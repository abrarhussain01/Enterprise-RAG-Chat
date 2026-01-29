from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader,TextLoader,Docx2txtLoader
from langchain_core.documents import Document
retriever = None
rag_chain = None
chat_history = []


MIN_TEXT_LENGTH = 30  # threshold to decide "real text"

def _validate_text(docs: list[Document], file_path: str):
    combined_text = " ".join(doc.page_content.strip() for doc in docs)

    if len(combined_text) < MIN_TEXT_LENGTH:
        raise ValueError(
            f"❌ ERROR: '{file_path}' appears to be image-based or contains no readable text."
        )


def _load_single_file(file: Path) -> list[Document]:
    suffix = file.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(file))
        docs = loader.load()
        _validate_text(docs, str(file))
        return docs

    elif suffix == ".txt":
        loader = TextLoader(str(file))
        return loader.load()

    elif suffix == ".docx":
        loader = Docx2txtLoader(str(file))
        docs = loader.load()
        _validate_text(docs, str(file))
        return docs

    else:
        return []




def load_documents(data_path: str) -> list[Document]:
    documents = []
    path = Path(data_path)

    # ---------- CASE 1: Single file ----------
    if path.is_file():
        documents.extend(_load_single_file(path))
        return documents

    # ---------- CASE 2: Directory ----------
    if path.is_dir():
        for file in path.iterdir():
            documents.extend(_load_single_file(file))
        return documents

    raise ValueError(f"Invalid path: {data_path}")


from langchain_text_splitters import RecursiveCharacterTextSplitter



def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_documents(documents)
    return chunks



from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def create_faiss_in_memory(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    return vectorstore



from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()


llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature=0.0,
    max_new_tokens=256,
    top_p=0.9,
    repetition_penalty=1.05
)

chat_model=ChatHuggingFace(llm=llm)



from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

def retrieve_context(question: str):
    docs = retriever.invoke(question)
    return "\n\n".join(doc.page_content for doc in docs)

def build_messages(inputs: dict):
    """
    inputs = {
        'question': str,
        'context': str
    }
    """
    SYSTEM_PROMPT = (
        "You are a precise and truthful assistant.\n"
        "Answer questions using ONLY the provided context.\n"
        "If the answer is not in the context, say:\n"
        "\"I don't know based on the provided documents.\""
    )
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    messages.extend(chat_history)
    messages.append(
        HumanMessage(
            content=f"Context:\n{inputs['context']}\n\nQuestion:\n{inputs['question']}"
        )
    )
    return messages

def save_ai_message(response):
    chat_history.append(AIMessage(content=response.content))
    return response.content

def build_rag_chain():
    global rag_chain

    parallel_inputs = RunnableParallel(
        question=RunnablePassthrough(),
        context=RunnableLambda(retrieve_context)
    )

    rag_chain = parallel_inputs | RunnableLambda(build_messages) | chat_model | RunnableLambda(save_ai_message)



def initialize_vector_store(pdf_path: str):
    global retriever, rag_chain, chat_history

    # reset state
    chat_history.clear()

    # 1. load docs
    docs = load_documents(pdf_path)

    # 2. split
    chunks = split_documents(docs)

    # 3. create vector store
    vect = create_faiss_in_memory(chunks)

    # 4. retriever
    retriever = vect.as_retriever(search_kwargs={"k": 3})

    # 5. build chain
    build_rag_chain()

    return True

def ask_question(question: str) -> str:
    if rag_chain is None:
        raise RuntimeError("Vector store not initialized")

    return rag_chain.invoke(question)

