import streamlit as st
import os
from rag import initialize_vector_store, ask_question

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(page_title="Enterprise RAG", layout="wide")

st.title("📄 Enterprise RAG Chat")

# ---------------- SESSION STATE ----------------

if "ready" not in st.session_state:
    st.session_state.ready = False

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- FILE UPLOAD ----------------

if not st.session_state.ready:
    st.subheader("Upload a document")

    uploaded_file = st.file_uploader(
        "Drag & drop or select a PDF",
        type=["pdf"]
    )

    if uploaded_file:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("Process Document"):
            with st.spinner("Building vector database..."):
                initialize_vector_store(file_path)
                st.session_state.ready = True
                st.success("Document processed. You can now chat.")

    #st.stop()

# ---------------- CHAT UI ----------------

st.subheader("Chat")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask a question about the document")

if query:
    st.session_state.messages.append(
        {"role": "user", "content": query}
    )

    with st.chat_message("user"):
        st.markdown(query)

    with st.spinner("Thinking..."):
        answer = ask_question(query)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    with st.chat_message("assistant"):
        st.markdown(answer)

# ---------------- EXIT BUTTON ----------------

if st.button("Exit Chat"):
    st.session_state.clear()
    st.rerun()
