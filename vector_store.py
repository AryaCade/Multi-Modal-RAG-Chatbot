from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pickle
import os

emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def build_faiss(chunks, index_path="vector_store"):
    """
    Create a FAISS store from normalized chunks.

    Args:
        chunks (list): List of normalized chunks.
        index_path (str): Path to save the FAISS index.

    Returns:
        FAISS: The created FAISS vector store.
    """

    # Extract text and metadata
    texts = []
    metadatas = []

    for ch in chunks:
        if not ch:
            continue

        # Safety: skip empty chunks
        content = ch.get("embedding_text", "") or ch.get("content", "")
        if not content.strip():
            continue

        texts.append(content)

        metadatas.append({
            "page": ch.get("page"),
            "type": ch.get("type"),
            "raw": ch.get("content"),
        })

    # Build a FAISS
    store = FAISS.from_texts(texts, embedding=emb, metadatas=metadatas)

    store.save_local(index_path)

    print(f"\nFAISS index built & saved to '{index_path}'")
    print(f"Total vectors: {len(texts)}")


    return store

def load_faiss(index_path="vector_store"):
    """
    Load existing FAISS index.

    Args:
        index_path (str): Path to the FAISS index.

    Returns:
        FAISS: The loaded FAISS vector store.
    """
    if not os.path.exists(index_path):
        raise RuntimeError("FAISS index not found!")
    
    store = FAISS.load_local(index_path, emb, allow_dangerous_deserialization=True)
    print(f"FAISS index loaded from '{index_path}'")
    return store
