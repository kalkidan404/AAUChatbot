
import streamlit as st
from rag_pipeline import get_answer



# --- Page config ---
st.set_page_config(page_title="AAU Assistant", layout="wide")

# --- Optimized Logic Loading ---
# This ensures your RAG model/docs load only once, saving memory and time
@st.cache_resource
def load_rag_logic():
    # If your get_answer function requires initialization, do it here
    return get_answer

rag_function = load_rag_logic()

st.title("🎓 AAU General Assistant")

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []

# User input
question = st.chat_input("Ask about Addis Ababa University...")

if question:
    st.session_state.history.append(("user", question))

    # DIRECT CALL to backend logic (No more localhost/requests needed!)
    try:
        # Assuming get_answer returns: answer, sources, chunks
        answer, sources, chunks = rag_function(question)
        
    except Exception as e:
        answer = f"Logic Error: {str(e)}"
        sources = []
        chunks = []

    st.session_state.history.append(("bot", answer))

    # --- Display chat history ---
    # Moved inside the 'if' or kept outside to refresh the view
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
    st.subheader("Sources")
    if sources:
        for s in sources:
            st.markdown(f'<p style="color: gold">{s}</p>', unsafe_allow_html=True)
    else:
        st.write("No sources found.")

    st.subheader("Retrieved Text")
    if chunks:
        for c in chunks:
            st.markdown(
                f'<div style="color: gold; background-color: black; padding: 5px; border-radius: 5px; margin-bottom: 5px">{c}</div>',
                unsafe_allow_html=True
            )
    else:
        st.write("No context retrieved.")
