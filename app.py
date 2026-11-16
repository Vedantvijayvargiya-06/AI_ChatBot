"""Smart Study Buddy - AI Study Assistant (Streamlit UI).

Run with:  streamlit run app.py
"""

import streamlit as st

from chatbot.assistant import StudyBuddyError, ask_question, get_api_key

st.set_page_config(
    page_title="Smart Study Buddy",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded",
)

SAMPLE_QUESTIONS = [
    "What is Python and why is it used?",
    "Explain Object-Oriented Programming with an example.",
    "What is Machine Learning?",
    "Explain the difference between AI and ML.",
    "What is a database and why do we need it?",
]

WELCOME = (
    "👋 Hi! I'm **Smart Study Buddy**. Ask me any academic question — Python, AI/ML, "
    "Data Structures, DBMS, Operating Systems, Computer Networks, Maths and more. "
    "Try one of the sample questions in the sidebar!"
)

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 820px;}
    .hero {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #db2777 100%);
        padding: 1.6rem 1.8rem; border-radius: 18px; color: #fff;
        box-shadow: 0 10px 30px rgba(79,70,229,.25); margin-bottom: 1.2rem;
    }
    .hero h1 {margin: 0; font-size: 2.1rem; font-weight: 800; letter-spacing: -.5px; color: #fff;}
    .hero p {margin: .35rem 0 0; font-size: 1.05rem; opacity: .92; color: #fff;}
    [data-testid="stChatMessage"] {
        border-radius: 16px; padding: .9rem 1rem; margin-bottom: .6rem;
        border: 1px solid rgba(128,128,128,.18);
        box-shadow: 0 2px 8px rgba(0,0,0,.05);
    }
    [data-testid="stSidebar"] .stButton button {
        width: 100%; text-align: left; border-radius: 10px;
    }
    .footer {
        text-align: center; opacity: .65; font-size: .85rem;
        margin-top: 2rem; padding-top: .8rem; border-top: 1px solid rgba(128,128,128,.25);
    }
    </style>
    <div class="hero">
        <h1>🎓 Smart Study Buddy</h1>
        <p>Your AI-powered personal study assistant.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role", "content", "error"(optional)}]
if "pending" not in st.session_state:
    st.session_state.pending = None  # question chosen from the sidebar


def clear_chat() -> None:
    st.session_state.messages = []
    st.session_state.pending = None


def use_sample(question: str) -> None:
    st.session_state.pending = question


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("📘 About")
    st.write(
        "**Smart Study Buddy** answers student questions with Google Gemini, "
        "orchestrated by LangChain (prompt template → Gemini → output parser)."
    )

    st.subheader("How to use")
    st.markdown(
        "1. Type a question in the box at the bottom.\n"
        "2. Press **Enter** to send.\n"
        "3. Ask follow-ups — I remember this session.\n"
        "4. Use **Clear chat** to start fresh."
    )

    st.subheader("💡 Try a sample question")
    for i, q in enumerate(SAMPLE_QUESTIONS):
        st.button(q, key=f"sample_{i}", on_click=use_sample, args=(q,))

    st.divider()
    st.button("🗑️ Clear chat", on_click=clear_chat, type="primary", use_container_width=True)

    st.caption("Built with Python • LangChain • Gemini")

# API-key status (the key itself is never displayed).
key_ok = True
try:
    get_api_key()
except StudyBuddyError as err:
    key_ok = False
    st.warning(f"🔑 {err.message}")

# ----------------------------------------------------------------------------
# Chat history
# ----------------------------------------------------------------------------
if not st.session_state.messages:
    with st.chat_message("assistant", avatar="🎓"):
        st.markdown(WELCOME)

for msg in st.session_state.messages:
    avatar = "🧑‍🎓" if msg["role"] == "user" else "🎓"
    with st.chat_message(msg["role"], avatar=avatar):
        if msg.get("error"):
            st.error(msg["content"])
        else:
            st.markdown(msg["content"])

# ----------------------------------------------------------------------------
# Handle a new question
# ----------------------------------------------------------------------------
typed = st.chat_input("Ask your study question here...")
question = st.session_state.pending or typed
st.session_state.pending = None

if question is not None:
    if not question.strip():
        st.warning("Please type a question before sending.")
    else:
        # Only successful turns are sent to the model as context.
        history = [m for m in st.session_state.messages if not m.get("error")]
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(question)

        with st.chat_message("assistant", avatar="🎓"):
            with st.spinner("Thinking... 🤔"):
                try:
                    answer = ask_question(question, history)
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except StudyBuddyError as err:
                    st.error(err.message)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": err.message, "error": True}
                    )
                except Exception:  # last-resort safety net: no stack traces for users
                    msg = "Unexpected error. Please try again."
                    st.error(msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": msg, "error": True}
                    )

st.markdown(
    '<div class="footer">Made with ❤️ using Python, LangChain &amp; Google Gemini</div>',
    unsafe_allow_html=True,
)
