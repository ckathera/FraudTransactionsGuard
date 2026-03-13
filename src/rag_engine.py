"""
RAG Engine — Fraud Policy Document Retrieval
Uses FAISS vector store + HuggingFace sentence-transformers (free, local, no API needed).
Documents: fraud patterns, investigation playbook, compliance rules.

Build index:  python src/rag_engine.py
"""
import os

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
POLICIES_DIR = os.path.join(BASE_DIR, "data", "fraud_rules")
VECTORSTORE_PATH = os.path.join(BASE_DIR, "data", "vectorstore")
EMBED_MODEL = "all-MiniLM-L6-v2"  # ~22MB, runs locally, no API key


def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def build_vectorstore() -> FAISS:
    """Load policy docs, split into chunks, build FAISS index."""
    policy_files = [f for f in os.listdir(POLICIES_DIR) if f.endswith(".txt")]
    docs = []
    for fname in policy_files:
        path = os.path.join(POLICIES_DIR, fname)
        loader = TextLoader(path, encoding="utf-8")
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60)
    chunks = splitter.split_documents(docs)

    embeddings = _get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)

    print(f"Built vectorstore: {len(chunks)} chunks from {len(policy_files)} documents")
    return vectorstore


def load_vectorstore() -> FAISS:
    """Load existing FAISS index, or build it if not found."""
    embeddings = _get_embeddings()
    if os.path.exists(VECTORSTORE_PATH):
        return FAISS.load_local(
            VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True
        )
    return build_vectorstore()


def retrieve(query: str, k: int = 4) -> str:
    """
    Retrieve the top-k most relevant policy chunks for a given query.
    Returns a single string of joined chunks, ready to inject into a prompt.
    """
    vs = load_vectorstore()
    docs = vs.similarity_search(query, k=k)
    return "\n\n---\n\n".join(d.page_content for d in docs)


if __name__ == "__main__":
    build_vectorstore()
    # Quick smoke test
    result = retrieve("high risk transaction block decision online fraud")
    print("\n── Sample retrieval result ──")
    print(result[:600])
