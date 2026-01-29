# 📄 Enterprise RAG Chat

An **enterprise‑grade Retrieval‑Augmented Generation (RAG) application** that allows users to upload documents (PDF/TXT/DOCX), build a vector database, and chat with an AI model grounded strictly on the uploaded content.

This project demonstrates **production‑style RAG architecture**, combining LangChain runnables, FAISS vector search, Hugging Face chat models, and a Streamlit UI with proper state management.

---

## 🚀 Features

* 📂 Drag‑and‑drop document upload (PDF supported, easily extensible)
* ⛔ Blocking ingestion phase (no queries allowed during indexing)
* 🧠 FAISS in‑memory vector database
* 🔁 Retriever invoked on every query (fresh context)
* 💬 Full conversational chat (entire chat history sent each turn)
* 🔗 Parallel + sequential LangChain runnable pipeline
* 🤖 Hugging Face **ChatModel** (instruction‑tuned)
* 🌐 Streamlit UI with clean session lifecycle
* 🧹 Exit button to reset chat and state

---

## 🧠 Architecture Overview

```
User (Streamlit UI)
   │
   ├── Upload Document
   │      └── Backend builds vector DB (blocking)
   │
   └── Ask Questions
          ├── Retriever (FAISS)
          ├── Full Chat History
          ├── Context + Question
          └── Chat LLM
```

### Key Design Principles

* **Retrieval is stateless** → always re‑computed per query
* **Conversation is stateful** → full chat history preserved
* **Grounded answers** → model answers only from retrieved context

---

## 📁 Project Structure

```
enterprise-rag/
│
├── app.py            # Streamlit UI
├── rag.py            # RAG backend logic
├── data/
│   └── uploads/      # Uploaded documents
├── .env              # Hugging Face token
├── requirements.txt
└── README.md
```

---

## 🔐 Environment Setup

⚠️ **Important:** You must use **your own Hugging Face API key** to run this application.

Create a `.env` file in the project root:

```env
HF_TOKEN=your_huggingface_token_here
```

> A **read‑only Hugging Face token** is sufficient.

---

## ▶️ Running the Application

```bash
streamlit run app.py
```

---

## 🧪 Application Flow

### 1️⃣ Upload Phase

* User uploads a document
* Backend loads, splits, embeds, and indexes the document
* Chat input is **disabled** during this phase

### 2️⃣ Chat Phase

* User asks questions about the document
* For **every query**:

  * Retriever fetches top‑K relevant chunks
  * Full chat history + context is sent to the LLM
  * AI response is generated and stored

### 3️⃣ Exit Phase

* User clicks **Exit Chat**
* Chat history and vector store state are cleared
* Application resets for a new document

---

## 🧠 RAG Pipeline Details

### Parallel + Sequential Chain

* **Parallel stage**:

  * Passes the question through
  * Retrieves relevant context

* **Sequential stage**:

  * Builds system + history + context prompt
  * Invokes Hugging Face ChatModel
  * Saves AI response to chat history

This ensures:

* High accuracy
* No hallucinations
* Proper conversational continuity

---

## ⚠️ Error Handling

* Empty or unsupported documents are rejected
* FAISS is never initialized with zero chunks
* User‑friendly errors shown in UI

---

## 🔮 Future Enhancements

* 📚 Multi‑document support
* 🔎 Source citation in responses
* 🧠 Query rewriting for follow‑ups
* 💾 Persistent vector storage
* 🖼️ OCR support for scanned PDFs

