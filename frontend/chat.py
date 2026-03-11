import streamlit as st
import requests

# --- Page config ---
st.set_page_config(page_title="AAU Assistant", layout="wide")

st.title("🎓 AAU General Assistant")

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []

# User input
question = st.chat_input("Ask about Addis Ababa University...")

if question:
    st.session_state.history.append(("user", question))

    # Send question to backend
    try:
        response = requests.post(
            "http://127.0.0.1:8000/ask",
            json={"question": question},
            timeout=10
        )
        data = response.json()
        answer = data.get("answer", "No answer returned.")
        sources = data.get("sources", [])
        chunks = data.get("chunks", [])

    except requests.exceptions.RequestException as e:
        answer = f"Error: {str(e)}"
        sources = []
        chunks = []

    st.session_state.history.append(("bot", answer))

# --- Display chat history ---
for role, message in st.session_state.history:
    if role == "user":
        st.chat_message("user").markdown(
            f'<p style="color: black; background-color: gold; padding: 5px; border-radius: 5px">{message}</p>',
            unsafe_allow_html=True
        )
    else:
        st.chat_message("assistant").markdown(
            f'<p style="color: gold; background-color: black; padding: 5px; border-radius: 5px">{message}</p>',
            unsafe_allow_html=True
        )

# --- Display sources and retrieved text ---
if question:
    st.subheader("Sources")
    for s in sources:
        st.markdown(f'<p style="color: gold">{s}</p>', unsafe_allow_html=True)

    st.subheader("Retrieved Text")
    for c in chunks:
        st.markdown(
            f'<div style="color: gold; background-color: black; padding: 5px; border-radius: 5px; margin-bottom: 5px">{c}</div>',
            unsafe_allow_html=True
        )