import streamlit as st

st.set_page_config(
    page_title="MediAI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── hide sidebar and default nav ──────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"]        { display: none !important; }
    [data-testid="collapsedControl"] { display: none !important; }
    header[data-testid="stHeader"]   { background: transparent; }
    .block-container { padding-top: 2rem; }

    .hero {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 60%, #3B82F6 100%);
        border-radius: 20px;
        padding: 56px 40px;
        text-align: center;
        color: white;
        margin-bottom: 36px;
    }
    .hero h1 { font-size: 52px; font-weight: 800; margin: 0; letter-spacing: -1.5px; color: white; }
    .hero p  { font-size: 18px; opacity: 0.85; margin: 14px 0 0; color: white; }

    .stat-box {
        background: white;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #E2E8F0;
        margin-bottom: 24px;
    }
    .stat-num   { font-size: 30px; font-weight: 800; color: #1E3A8A; }
    .stat-label { font-size: 13px; color: #64748B; margin-top: 4px; }

    .feature-card {
        background: white;
        border-radius: 16px;
        padding: 28px 24px;
        border: 1px solid #E2E8F0;
        min-height: 180px;
        margin-bottom: 12px;
    }
    .feature-icon  { font-size: 36px; margin-bottom: 10px; }
    .feature-title { font-size: 16px; font-weight: 700; color: #1E293B; margin-bottom: 6px; }
    .feature-desc  { font-size: 13px; color: #64748B; line-height: 1.6; }

    .about-box {
        background: white;
        border-radius: 16px;
        padding: 28px;
        border: 1px solid #E2E8F0;
        margin-bottom: 24px;
    }
    .about-title { font-size: 15px; font-weight: 700; color: #1E293B; margin-bottom: 14px; }

    .warning-bar {
        background: #FEF3C7;
        border: 1px solid #FDE68A;
        border-radius: 10px;
        padding: 12px 20px;
        font-size: 13px;
        color: #92400E;
        text-align: center;
        margin-top: 8px;
    }

    div[data-testid="stButton"] > button {
        background: #2563EB !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 10px 16px !important;
        width: 100% !important;
        transition: background 0.2s !important;
    }
    div[data-testid="stButton"] > button:hover {
        background: #1D4ED8 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── clear chat state when on home page ────────────────────────
for key in ["messages", "quick_q"]:
    if key in st.session_state:
        del st.session_state[key]

# ── hero ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div style="font-size:24px;margin-bottom:10px">🏥</div>
    <h1>MediAI</h1>
    <p>AI-powered health tools — disease prediction, medical imaging & intelligent health chat</p>
</div>
""", unsafe_allow_html=True)

# ── stats ─────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="stat-box"><div class="stat-num">3</div><div class="stat-label">AI Models</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="stat-box"><div class="stat-num">90%</div><div class="stat-label">X-Ray Accuracy</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="stat-box"><div class="stat-num">106K+</div><div class="stat-label">Patients Trained On</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="stat-box"><div class="stat-num">4</div><div class="stat-label">Tools Available</div></div>', unsafe_allow_html=True)

# ── feature cards ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 🚀 Choose a Tool")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🔬</div>
        <div class="feature-title">Diabetes Risk</div>
        <div class="feature-desc">Predict diabetes risk from 8 health metrics including glucose, BMI, and age.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Diabetes Tool", key="b1"):
        st.switch_page("pages/1_predict.py")

with c2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">❤️</div>
        <div class="feature-title">Heart Disease Risk</div>
        <div class="feature-desc">Predict heart disease risk from cardiac metrics including ECG and cholesterol.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Heart Tool", key="b2"):
        st.switch_page("pages/2_heart.py")

with c3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">💬</div>
        <div class="feature-title">Health Chat</div>
        <div class="feature-desc">Ask health questions in plain English. Powered by Llama3 running locally.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Health Chat", key="b3"):
        st.switch_page("pages/3_chat.py")

with c4:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🩺</div>
        <div class="feature-title">X-Ray Analyser</div>
        <div class="feature-desc">Upload a chest X-ray and AI detects pneumonia with 90% accuracy.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open X-Ray Tool", key="b4"):
        st.switch_page("pages/4_imaging.py")

# ── about ─────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
c1, c2 = st.columns(2)

with c1:
    st.markdown("""
    <div class="about-box">
        <div class="about-title">🛠️ Built With</div>
        <table style="width:100%;font-size:13px;color:#475569">
            <tr><td>🔥 PyTorch</td><td>Neural network models</td></tr>
            <tr><td>🌐 Streamlit</td><td>Web interface</td></tr>
            <tr><td>🦙 Llama3</td><td>AI chat assistant</td></tr>
            <tr><td>🖼️ ResNet18</td><td>Chest X-ray analysis</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="about-box">
        <div class="about-title">📊 Trained On</div>
        <table style="width:100%;font-size:13px;color:#475569">
            <tr><td>🩸 Pima Diabetes</td><td>768 patients</td></tr>
            <tr><td>❤️ Cleveland Heart</td><td>297 patients</td></tr>
            <tr><td>🫁 Chest X-Ray</td><td>5,216 images</td></tr>
            <tr><td>💊 Hospital Records</td><td>100,000+ patients</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="warning-bar">⚠️ MediAI is for educational purposes only. Never use AI predictions as a substitute for professional medical advice.</div>', unsafe_allow_html=True)