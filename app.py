import os
import uuid
import streamlit as st
import boto3
import requests
import json
import base64
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

# --- AWS CONFIG (values from .env; see .env.example) ---
REGION = os.environ["AWS_REGION"]
DYNAMODB_TABLE_NAME = os.environ["DYNAMODB_TABLE_NAME"]
API_GATEWAY_URL = os.environ["API_GATEWAY_URL"]  # e.g. https://xxxx.execute-api.us-east-1.amazonaws.com/prod/analyze

dynamodb = boto3.resource("dynamodb", region_name=REGION)
table = dynamodb.Table(DYNAMODB_TABLE_NAME)

# One id per browser session so DynamoDB can track per-user cumulative totals
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

st.set_page_config(page_title="Pure-Cycle AI", page_icon="🌊", layout="wide")

# --- UI ---
st.title("🌊 Pure-Cycle: Ghost Fiber Analyzer")
st.markdown("### For SITA Sustainability Hackathon")

with st.expander("Problem statement & Why this matters", expanded=True):
    st.markdown(
        """
        **What's the Problem?**
        Synthetic textiles shed microfibers during washing. A significant share of ocean microplastics comes from textiles, but most people
        have no simple way to understand a garment’s shedding risk from the care label.

        **Why Now?**
        Small behavior changes (cold wash, gentler cycle, fiber capture) can meaningfully reduce microfiber pollution—if guidance is fast,
        practical, and personalized to what the garment is made of.

        **What Pure-Cycle App Does...**
        Upload a care-label photo to estimate “Ocean Impact”, get 2 easy washing tips, and see an estimated microplastics reduction.
        """
    )

# Sidebar Leaderboard
with st.sidebar:
    st.header("Community Stats")
    try:
        # Simple scan counter
        res = table.scan(Select='COUNT')
        st.metric("Total Items Scanned", res['Count'])
    except:
        st.metric("Total Items Scanned", "0")

# Main Interface
uploaded_file = st.file_uploader("Snap a photo of your clothing's care label", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Label for Analysis", width=350)
    
    # Image Preparation
    MAX_EDGE_PX = 1568
    #Anthropic recommends to resize image pixel by pixel to 1568 to avoid token limit issues
    buffered = BytesIO()
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")

    w, h = img.size
    scale = min(1.0, MAX_EDGE_PX / max(w, h))
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    img.save(buffered, format="WEBP", quality=80, method=6)
    img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    st.write(f"DEBUG: img_b64 length = {len(img_b64)}, starts with: {img_b64[:30]}")
    st.write(f"DEBUG: payload size = {len(img_b64.encode('utf-8'))} bytes")

    if st.button("Analyze Ocean Impact"):
        with st.spinner("Calling Pure-Cycle API..."):
            try:
                # Call API Gateway -> Lambda (Lambda handles S3 upload,
                # Bedrock call, microplastic calc, and DynamoDB save)
                response = requests.post(
                    API_GATEWAY_URL,
                    json={
                        "user_id": st.session_state.user_id,
                        "image_base64": img_b64,
                        "media_type": "image/webp",
                    },
                    timeout=60,
                )
                st.write(f"DEBUG: status_code = {response.status_code}")
                st.write(f"DEBUG: response.text = {response.text}")
                response.raise_for_status()
                result = response.json()

                data = result["analysis"]
                impact = result["impact"]
                cumulative_reduction_mg = result.get("cumulative_reduction_mg", 0)

                materials = str(data.get("materials", "")).strip()
                score = data.get("score", None)
                why_it_matters = str(data.get("why_it_matters", "")).strip()
                estimated_reduction = str(data.get("estimated_impact_reduction", "")).strip()
                wash_settings = data.get("recommended_wash_settings", []) or []
                actions = data.get("microplastic_reduction_actions", []) or []

                # Display Result
                st.divider()
                c1, c2, c3 = st.columns(3)
                c1.metric("Ocean Impact Score", f"{score}/10" if score is not None else "—")
                c2.info(f"**Composition:** {materials or '—'}")
                if estimated_reduction:
                    c3.success(f"**Estimated impact reduction:** {estimated_reduction}")

                m1, m2 = st.columns(2)
                m1.metric("Microplastics Reduced (this scan)", f"{impact['reduction_mg']} mg")
                m2.metric("Your Cumulative Reduction", f"{cumulative_reduction_mg} mg")

                st.subheader("Why It Matters")
                st.write(why_it_matters or "—")

                st.subheader("Recommended Wash Settings")
                if wash_settings:
                    for s in wash_settings:
                        st.write(f"🔹 {s}")
                else:
                    st.write("—")

                st.subheader("Microplastic Reduction Actions")
                if actions:
                    for a in actions:
                        st.write(f"🔹 {a}")
                else:
                    st.write("—")

                with st.expander("Raw API Response", expanded=False):
                    st.code(json.dumps(result, indent=2))

            except requests.exceptions.RequestException as e:
                st.error(f"API call failed: {str(e)}")
            except (KeyError, ValueError) as e:
                st.error(f"Unexpected API response: {str(e)}")