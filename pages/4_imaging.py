import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

st.set_page_config(page_title="X-Ray Analyser — MediAI", page_icon="🩺", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .page-header { margin-bottom: 28px; }
    .page-header h2 { font-size: 28px; font-weight: 700; color: #1E293B; margin: 0; }
    .page-header p  { font-size: 14px; color: #CBD5E1; margin: 6px 0 0; }
    .result-card { border-radius: 16px; padding: 28px; text-align: center; }
    .result-high { background: #FEF2F2; border: 1.5px solid #FCA5A5; }
    .result-low  { background: #F0FDF4; border: 1.5px solid #86EFAC; }
    .result-title { font-size: 22px; font-weight: 700; margin-bottom: 8px; }
    .result-prob  { font-size: 48px; font-weight: 800; margin: 12px 0; }
    .result-desc  { font-size: 14px; color: #475569; }
    .upload-box {
        background: #F8FAFC;
        border: 2px dashed #CBD5E1;
        border-radius: 16px;
        padding: 40px;
        text-align: center;
    }
    .info-box {
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 12px;
        padding: 16px 20px;
        font-size: 13px;
        color: #1E40AF;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ── load model (cached) ───────────────────────────────────────
@st.cache_resource
def load_model():
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 2)
    m.load_state_dict(torch.load(
        'models/saved/xray_model.pth',
        map_location='cpu',
        weights_only=True
    ))
    m.eval()
    return m

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ── header ────────────────────────────────────────────────────
if st.button("← Back to MediAI"):
    st.switch_page("app.py")

st.markdown("""
<div class="page-header">
    <h2>🩺 Chest X-Ray Analyser</h2>
    <p>Upload a chest X-ray image and AI will detect signs of pneumonia with 90% accuracy.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    ℹ️ This model was trained on 5,216 chest X-ray images using ResNet18 with transfer learning.
    Supported formats: JPG, JPEG, PNG
</div>
""", unsafe_allow_html=True)

# ── layout ────────────────────────────────────────────────────
col_upload, col_result = st.columns([1, 1], gap="large")

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload Chest X-Ray",
        type=['jpg', 'jpeg', 'png'],
        label_visibility="collapsed"
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded X-Ray", use_column_width=True)
    else:
        st.markdown("""
        <div class="upload-box">
            <div style="font-size:48px">🫁</div>
            <div style="font-size:15px;color:#64748B;margin-top:12px;font-weight:500">Drop your X-ray here</div>
            <div style="font-size:13px;color:#94A3B8;margin-top:6px">JPG, JPEG, or PNG</div>
        </div>
        """, unsafe_allow_html=True)

with col_result:
    st.markdown('<div style="font-size:14px;font-weight:600;color:#64748B;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:16px">Analysis Result</div>', unsafe_allow_html=True)

    if uploaded_file:
        with st.spinner("Analysing X-Ray with AI..."):
            img_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                output      = model(img_tensor)
                probs       = torch.softmax(output, dim=1)
                normal_prob = probs[0][0].item()
                pneumo_prob = probs[0][1].item()
                prediction  = output.argmax(1).item()

        if prediction == 0:
            st.markdown(f"""
            <div class="result-card result-low">
                <div class="result-title" style="color:#16A34A">✅ Normal</div>
                <div class="result-prob" style="color:#16A34A">{normal_prob*100:.1f}%</div>
                <div class="result-desc">No signs of pneumonia detected in this X-ray.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card result-high">
                <div class="result-title" style="color:#DC2626">⚠️ Pneumonia Detected</div>
                <div class="result-prob" style="color:#DC2626">{pneumo_prob*100:.1f}%</div>
                <div class="result-desc">Signs of pneumonia found. Please consult a doctor immediately.</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Confidence breakdown**")

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Normal", f"{normal_prob*100:.1f}%")
            st.progress(normal_prob)
        with c2:
            st.metric("Pneumonia", f"{pneumo_prob*100:.1f}%")
            st.progress(pneumo_prob)

        st.markdown("<br>", unsafe_allow_html=True)
        if prediction == 1:
            st.markdown("""
            **Recommended next steps:**
            - Consult a pulmonologist immediately
            - Get a follow-up CT scan
            - Start prescribed antibiotics if confirmed
            """)
        else:
            st.markdown("""
            **Looking good:**
            - Continue regular health check-ups
            - Maintain good respiratory hygiene
            - See a doctor if symptoms persist
            """)
    else:
        st.markdown("""
        <div style="background:#F8FAFC;border-radius:16px;padding:60px 40px;text-align:center;border:1px dashed #CBD5E1">
            <div style="font-size:40px">🩺</div>
            <div style="font-size:15px;color:#64748B;margin-top:12px">Upload a chest X-ray on the left to see the AI analysis here.</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.caption("⚠️ For educational purposes only. Always consult a qualified radiologist and medical professional.")