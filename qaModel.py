import fitz
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA


def load_pdf_text(file_path):
    """Load PDF text with validation"""
    with fitz.open(file_path) as doc:
        text = "\n".join(page.get_text() for page in doc)
        if not text.strip():
            raise ValueError("PDF contains no readable content")
        return text


def create_faiss_index(text):
    """Create FAISS index with version-compatible parameters"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=128,
        length_function=len
    )
    docs = splitter.create_documents([text])

    return FAISS.from_documents(
        documents=docs,
        embedding=HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
    )


def initialize_qa_system(db):
    """Initialize QA system with CPU optimization"""
    llm = CTransformers(
        model="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        model_type="mistral",
        config={
            'max_new_tokens': 512,
            'temperature': 0.3,
            'context_length': 2048,
            'threads': 8  # Use 8 CPU threads
        }
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        chain_type="stuff",
        return_source_documents=True
    )
