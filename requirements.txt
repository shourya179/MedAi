import streamlit as st
from groq import Groq

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

@st.cache_resource
def get_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])

client = get_client()

if st.button("← Back to MediAI"):
    st.switch_page("app.py")

st.markdown("""
<div class="page-header">
    <h2>💬 AI Health Assistant</h2>
    <p>Ask health questions in plain English. Powered by Llama3 via Groq.</p>
</div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "quick_q" not in st.session_state:
    st.session_state.quick_q = None

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

if st.session_state.quick_q:
    user_input = st.session_state.quick_q
    st.session_state.quick_q = None
else:
    user_input = None

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

typed_input = st.chat_input("Ask a health question...")
if typed_input:
    user_input = typed_input

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in st.session_state.messages:
        full_messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_reply  = ""

        with st.spinner("Thinking..."):
            stream = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=full_messages,
                stream=True
            )

        for chunk in stream:
            word        = chunk.choices[0].delta.content or ""
            full_reply += word
            placeholder.markdown(full_reply + "▌")

        placeholder.markdown(full_reply)

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_reply
    })

if st.session_state.messages:
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()

st.markdown("<br>", unsafe_allow_html=True)
st.caption("⚠️ For educational purposes only. Always consult a real doctor.")