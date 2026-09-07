import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pickle

st.set_page_config(page_title="Heart Disease Risk — MediAI", page_icon="❤️", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .page-header { margin-bottom: 28px; }
    .page-header h2 { font-size: 28px; font-weight: 700; color: #1E293B; margin: 0; }
    .page-header p  { font-size: 14px; color: #CBD5E1; margin: 6px 0 0; }
    .result-card { border-radius: 16px; padding: 28px; text-align: center; margin-top: 20px; }
    .result-high { background: #FEF2F2; border: 1.5px solid #FCA5A5; }
    .result-low  { background: #F0FDF4; border: 1.5px solid #86EFAC; }
    .result-title { font-size: 22px; font-weight: 700; margin-bottom: 8px; }
    .result-prob  { font-size: 48px; font-weight: 800; margin: 12px 0; }
    .result-desc  { font-size: 14px; color: #475569; }
    .input-section { background: white; border-radius: 16px; padding: 28px; border: 1px solid #E2E8F0; margin-bottom: 20px; }
    .section-title { font-size: 14px; font-weight: 600; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 16px; }
</style>
""", unsafe_allow_html=True)

# ── load model (cached) ───────────────────────────────────────
@st.cache_resource
def load_model():
    class HeartNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(13, 32), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(32, 16), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(16, 8),  nn.ReLU(),
                nn.Linear(8, 1)
            )
        def forward(self, x):
            return self.network(x)

    m = HeartNet()
    m.load_state_dict(torch.load('models/saved/heart_model.pth', map_location='cpu', weights_only=True))
    m.eval()
    return m

@st.cache_resource
def load_scaler():
    with open('models/saved/heart_scaler.pkl', 'rb') as f:
        return pickle.load(f)

model  = load_model()
scaler = load_scaler()

# ── header ────────────────────────────────────────────────────
if st.button("← Back to MediAI"):
    st.switch_page("app.py")

st.markdown("""
<div class="page-header">
    <h2>❤️ Heart Disease Risk Predictor</h2>
    <p>Enter your cardiac health metrics to assess your heart disease risk.</p>
</div>
""", unsafe_allow_html=True)

# ── form ──────────────────────────────────────────────────────
col_form, col_result = st.columns([1.2, 1], gap="large")

with col_form:
    st.markdown('<div class="section-title">Personal Information</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", 1, 120, 45)
        sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    with c2:
        cp  = st.selectbox("Chest Pain Type", [1, 2, 3, 4],
                           format_func=lambda x: {1:"Typical Angina", 2:"Atypical Angina", 3:"Non-anginal", 4:"Asymptomatic"}[x])
        fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1],
                           format_func=lambda x: "No" if x == 0 else "Yes")

    st.markdown('<div class="section-title" style="margin-top:20px">Clinical Measurements</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
        chol     = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
        restecg  = st.selectbox("Resting ECG", [0, 1, 2])
        thalach  = st.number_input("Max Heart Rate", 60, 220, 150)
    with c2:
        exang   = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        oldpeak = st.number_input("ST Depression", 0.0, 7.0, 1.0, step=0.1)
        slope   = st.selectbox("Slope of ST Segment", [1, 2, 3])
        ca      = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])

    thal = st.selectbox("Thalassemia", [3, 6, 7],
                        format_func=lambda x: {3:"Normal", 6:"Fixed Defect", 7:"Reversible Defect"}[x])
    
    predict_btn = st.button("❤️ Predict Heart Disease Risk", type="primary", use_container_width=True)

# ── result ────────────────────────────────────────────────────
with col_result:
    st.markdown('<div class="section-title">Result</div>', unsafe_allow_html=True)

    if predict_btn:
        input_data   = np.array([[age, sex, cp, trestbps, chol, fbs,
                                   restecg, thalach, exang, oldpeak, slope, ca, thal]])
        input_scaled = scaler.transform(input_data)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

        with torch.no_grad():
            output      = model(input_tensor)
            probability = torch.sigmoid(output).item()
            prediction  = 1 if probability >= 0.5 else 0

        if prediction == 1:
            st.markdown(f"""
            <div class="result-card result-high">
                <div class="result-title" style="color:#DC2626">⚠️ High Risk</div>
                <div class="result-prob" style="color:#DC2626">{probability*100:.1f}%</div>
                <div class="result-desc">Heart disease risk detected. Please consult a cardiologist immediately.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card result-low">
                <div class="result-title" style="color:#16A34A">✅ Low Risk</div>
                <div class="result-prob" style="color:#16A34A">{probability*100:.1f}%</div>
                <div class="result-desc">No significant heart disease risk detected. Keep your heart healthy!</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Risk level**")
        st.progress(probability)

        st.markdown("<br>", unsafe_allow_html=True)
        if prediction == 1:
            st.markdown("""
            **Recommended next steps:**
            - See a cardiologist urgently
            - Get an ECG and stress test
            - Monitor blood pressure daily
            """)
        else:
            st.markdown("""
            **Keep it up:**
            - Exercise regularly
            - Maintain healthy cholesterol
            - Avoid smoking and excess alcohol
            """)
    else:
        st.markdown("""
        <div style="background:#F8FAFC;border-radius:16px;padding:40px;text-align:center;border:1px dashed #CBD5E1">
            <div style="font-size:40px">❤️</div>
            <div style="font-size:15px;color:#64748B;margin-top:12px">Fill in your cardiac metrics and click Predict to see your heart disease risk score.</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.caption("⚠️ This is an AI prediction tool for educational purposes only. Always consult a qualified medical professional.")