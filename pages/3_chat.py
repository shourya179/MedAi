import streamlit as st
import google.generativeai as genai

st.set_page_config(page_title="Health Chat — MediAI", page_icon="💬", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"]        { display: none !important; }
    [data-testid="collapsedControl"] { display: none !important; }
    .block-container { padding-top: 2rem; }
    .page-header h2 { font-size: 28px; font-weight: 700; color: #FFFFFF; margin: 0; }
    .page-header p  { font-size: 14px; color: #CBD5E1; margin: 6px 0 0; }
    .quick-label    { font-size: 13px; font-weight: 600; color: #94A3B8; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

SYSTEM_PROMPT = """You are MediAI, a helpful and empathetic AI health assistant.
You help users understand health conditions, symptoms, and general wellness advice.
Always recommend consulting a real doctor for diagnosis and treatment.
Keep responses clear, concise and easy to understand.
Never diagnose — only educate and inform."""

# ── configure Gemini ──────────────────────────────────────────
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

@st.cache_resource
def get_model():
    return genai.GenerativeModel(
        model_name="gemini-1.5-flash-latest",
        system_instruction=SYSTEM_PROMPT
    )

model = get_model()

# ── header ────────────────────────────────────────────────────
if st.button("← Back to MediAI"):
    st.switch_page("app.py")

st.markdown("""
<div class="page-header">
    <h2>💬 AI Health Assistant</h2>
    <p>Ask health questions in plain English. Powered by Gemini AI.</p>
</div>
""", unsafe_allow_html=True)

# ── session state ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_session" not in st.session_state:
    st.session_state.chat_session = model.start_chat(history=[])
if "quick_q" not in st.session_state:
    st.session_state.quick_q = None

# ── quick questions ───────────────────────────────────────────
st.markdown('<div class="quick-label">Quick questions</div>', unsafe_allow_html=True)

quick_questions = [
    "Symptoms of diabetes?",
    "How to lower blood pressure?",
    "What causes chest pain?",
    "How to improve heart health?",
    "What is a healthy BMI?",
]

cols = st.columns(len(quick_questions))
for i, q in enumerate(quick_questions):
    with cols[i]:
        if st.button(q, use_container_width=True, key=f"qq_{i}"):
            st.session_state.quick_q = q

st.markdown("---")

# ── pick up quick question ────────────────────────────────────
if st.session_state.quick_q:
    user_input = st.session_state.quick_q
    st.session_state.quick_q = None
else:
    user_input = None

# ── display chat history ──────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── typed input ───────────────────────────────────────────────
typed_input = st.chat_input("Ask a health question...")
if typed_input:
    user_input = typed_input

# ── process input ─────────────────────────────────────────────
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_reply  = ""

        try:
            response = st.session_state.chat_session.send_message(
                user_input,
                stream=True
            )

            for chunk in response:
                full_reply += chunk.text
                placeholder.markdown(full_reply + "▌")

            placeholder.markdown(full_reply)

        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_reply
    })

# ── clear button ──────────────────────────────────────────────
if st.session_state.messages:
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.session_state.chat_session = model.start_chat(history=[])
        st.rerun()

st.markdown("<br>", unsafe_allow_html=True)
st.caption("⚠️ For educational purposes only. Always consult a real doctor.")