"""
Pure-Cycle: Ghost Fiber Analyzer - Lambda handler.

Flow per request:
  1. Decode the incoming base64 image and store it in S3.
  2. Call Bedrock (Claude) with the image to get a washing recommendation.
  3. Estimate the microplastic reduction (mg) implied by that recommendation.
  4. Persist the scan + running per-user cumulative total in DynamoDB.

Expects to be invoked via API Gateway (Lambda proxy integration), with a
JSON body of the form:
    {
      "user_id": "abc123",
      "image_base64": "<base64 webp/jpeg/png bytes>",
      "media_type": "image/webp"   # optional, defaults to image/webp
    }

Environment variables (set in Lambda console -> Configuration):
    S3_BUCKET_NAME       - bucket to store uploaded label images
    TABLE_NAME           - DynamoDB table for scan history + cumulative totals
    BEDROCK_MODEL_ID      - optional, defaults to Claude Haiku on Bedrock
    AWS_REGION_NAME       - optional, defaults to the Lambda's own region
"""

import base64
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal

import boto3

# --- Config from environment ---
S3_BUCKET_NAME = os.environ["S3_BUCKET_NAME"]
TABLE_NAME = os.environ["TABLE_NAME"]
BEDROCK_MODEL_ID = os.environ.get(
    "BEDROCK_MODEL_ID"
)
REGION = os.environ.get("AWS_REGION_NAME") or os.environ.get("AWS_REGION")

s3 = boto3.client("s3", region_name=REGION)
bedrock = boto3.client("bedrock-runtime", region_name=REGION)
dynamodb = boto3.resource("dynamodb", region_name=REGION)
table = dynamodb.Table(TABLE_NAME)

# Rough baseline: fully-synthetic garment sheds ~900mg of microfibers per
# wash (mid-point of commonly cited 600-900mg/wash research figures for
# synthetic textiles). We scale that baseline by the Ocean Impact Score
# (1-10) to approximate how synthetic-heavy a given garment is.
MAX_BASELINE_SHED_MG = 900.0

PROMPT_TEXT = """
<instructions>
You are a Marine Biologist and Textile Scientist. You specialize in microfiber pollution from textiles and evidence-based washing practices that reduce microplastic runoff into aquatic ecosystems.

Base your recommendations on established environmental research regarding microfiber shedding, including factors such as material type (synthetic vs natural), water temperature, agitation level, and spin speed.
</instructions>
<task>
1. Identify all materials listed on the clothing label and their percentages.

2. Classify each material by microplastic shedding risk:
   - High Risk: polyester, nylon, acrylic, elastane/spandex, synthetic blends
   - Medium Risk: rayon/viscose and partial synthetic blends
   - Low Risk: cotton, linen, wool

3. Provide an 'Ocean Impact Score' (1-10), where:
   - 1-3 = low (mostly natural fibers)
   - 4-6 = mixed/blended materials
   - 7-10 = high synthetic content and high shedding potential

4. Generate 2-3 practical washing recommendations that directly reduce microplastic runoff.
   These must:
   - Be easy to follow in a single wash (no multi-step routines)
   - Include specific machine settings where relevant:
     - Cold water (<=30C / 86F)
     - Gentle/Delicate cycle
     - Low to medium spin speed
   - Include at least one microplastic-specific action when synthetics are present (e.g., use of a microfiber bag or filter, washing less frequently)

5. Provide an estimated reduction in microplastic shedding (as a percentage range) if the recommendations are followed.
   - Base estimates on general research trends (e.g., colder water and reduced agitation can reduce shedding by ~20-50%)
   - Adjust estimates depending on material risk level

6. Keep tone action-oriented and informative, not judgmental or alarmist.

7. Output format:
Return ONLY valid JSON. Do not wrap in markdown fences.

Schema:
{
  "materials": "string (all materials + percentages as written)",
  "material_risk": [{"material": "string", "risk": "High|Medium|Low"}],
  "score": 1-10,
  "why_it_matters": "string (1-2 sentences)",
  "recommended_wash_settings": ["string", "string", "string"],
  "microplastic_reduction_actions": ["string", "string"],
  "estimated_impact_reduction": "string (e.g. \\"35-50%\\")"
}
</task>
"""


class AnalysisError(Exception):
    pass


def _cors_response(status_code, payload):
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(payload, default=str),
    }


def _extract_json(model_text):
    cleaned = (model_text or "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise AnalysisError(f"Model returned non-JSON output: {cleaned[:500]}")


def analyze_with_bedrock(image_b64, media_type):
    """Step 2: call Bedrock with the label image and parse the JSON result."""
    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": PROMPT_TEXT},
                    ],
                }
            ],
        }
    )

    response = bedrock.invoke_model(modelId=BEDROCK_MODEL_ID, body=body)
    raw_body = response.get("body").read()
    if not raw_body:
        raise AnalysisError("Bedrock returned an empty response body.")

    result = json.loads(raw_body.decode("utf-8", errors="replace"))
    model_text = ""
    try:
        model_text = result["content"][0].get("text", "")
    except (KeyError, IndexError, AttributeError):
        pass

    return _extract_json(model_text)


def _parse_reduction_pct(estimated_impact_reduction):
    """Parse '35-50%' / '35–50%' / '40%' into a 0-100 midpoint value."""
    if not estimated_impact_reduction:
        return 0.0
    numbers = re.findall(r"\d+(?:\.\d+)?", str(estimated_impact_reduction))
    if not numbers:
        return 0.0
    values = [float(n) for n in numbers[:2]]
    return sum(values) / len(values)


def calculate_microplastic_reduction(data):
    """Step 3: estimate mg of microplastics saved by following the advice."""
    try:
        score = float(data.get("score", 0))
    except (TypeError, ValueError):
        score = 0.0
    score = max(0.0, min(10.0, score))

    baseline_shed_mg = (score / 10.0) * MAX_BASELINE_SHED_MG
    reduction_pct = _parse_reduction_pct(data.get("estimated_impact_reduction"))
    reduction_mg = baseline_shed_mg * (reduction_pct / 100.0)

    return {
        "baseline_shed_mg": round(baseline_shed_mg, 1),
        "reduction_pct": round(reduction_pct, 1),
        "reduction_mg": round(reduction_mg, 1),
    }


def save_scan_and_update_totals(user_id, scan_id, timestamp, data, impact):
    """Step 4: persist this scan and atomically update the user's running total."""
    table.put_item(
        Item={
            "user_id": user_id,
            "scan_id": scan_id,
            "date": timestamp,
            "materials": str(data.get("materials", "")).strip() or "Unknown",
            "score": Decimal(str(data.get("score", -1))),
            "why_it_matters": str(data.get("why_it_matters", "")).strip(),
            "recommended_wash_settings": data.get("recommended_wash_settings", []) or [],
            "microplastic_reduction_actions": data.get("microplastic_reduction_actions", []) or [],
            "estimated_impact_reduction": str(data.get("estimated_impact_reduction", "")).strip(),
            "baseline_shed_mg": Decimal(str(impact["baseline_shed_mg"])),
            "reduction_mg": Decimal(str(impact["reduction_mg"])),
        }
    )

    summary = table.update_item(
        Key={"user_id": user_id, "scan_id": "SUMMARY"},
        UpdateExpression=(
            "ADD cumulative_reduction_mg :r, scan_count :one "
            "SET last_updated = :t"
        ),
        ExpressionAttributeValues={
            ":r": Decimal(str(impact["reduction_mg"])),
            ":one": 1,
            ":t": timestamp,
        },
        ReturnValues="ALL_NEW",
    )
    return summary["Attributes"]


def lambda_handler(event, context):
    try:
        raw_body = event.get("body", "{}") or "{}"
        if event.get("isBase64Encoded"):
            raw_body = base64.b64decode(raw_body).decode("utf-8")
        payload = json.loads(raw_body)

        user_id = str(payload.get("user_id") or "anonymous")
        image_b64 = payload.get("image_base64")
        media_type = payload.get("media_type", "image/webp")
        if not image_b64:
            return _cors_response(400, {"error": "image_base64 is required"})

        image_bytes = base64.b64decode(image_b64)

        timestamp = datetime.now(timezone.utc).isoformat()
        scan_id = f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
        ext = media_type.split("/")[-1] if "/" in media_type else "webp"
        s3_key = f"uploads/{user_id}/{scan_id}.{ext}"

        # 1. Save image to S3
        s3.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_key,
            Body=image_bytes,
            ContentType=media_type,
        )

        # 2. Call Bedrock for the washing recommendation
        data = analyze_with_bedrock(image_b64, media_type)

        # 3. Calculate microplastic reduction
        impact = calculate_microplastic_reduction(data)

        # 4. Save scan + cumulative total to DynamoDB
        summary = save_scan_and_update_totals(user_id, scan_id, timestamp, data, impact)

        return _cors_response(
            200,
            {
                "scan_id": scan_id,
                "s3_key": s3_key,
                "analysis": data,
                "impact": impact,
                "cumulative_reduction_mg": float(summary.get("cumulative_reduction_mg", 0)),
                "scan_count": int(summary.get("scan_count", 0)),
            },
        )

    except AnalysisError as e:
        return _cors_response(502, {"error": str(e)})
    except Exception as e:  # noqa: BLE001 - surface any failure to the client
        return _cors_response(500, {"error": f"Analysis failed: {e}"})
