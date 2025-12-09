from vector_store import load_faiss

# Load previously saved FAISS index
store = load_faiss("vector_store_mm")

def retrieve_chunks(query, store, k=5):
    """
    Search FAISS index and return top-k chunks and metadata.

    Args:
        query (str): The search query.
        store (FAISS): The FAISS vector store.
        k (int): Number of top results to return.

    Returns:
        tuple: (list of chunk texts, list of metadata dictionaries)
    """

    # Perform FAISS vector search
    result = store.similarity_search(query, k=k)

    contexts = []
    metadata = []

    for r in result:

        contexts.append(r.page_content)
        metadata.append(r.metadata)

    return contexts, metadata
