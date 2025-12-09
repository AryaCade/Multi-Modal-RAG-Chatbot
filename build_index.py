from multi_modal_ingest import multi_modal_ingest, normalize_element
from vector_store import build_faiss

import json

PDF_PATH = "multi-modal_rag_qa_assignment.pdf"

def load_pdf_bytes(path):
    with open(path, "rb") as f:
        return f.read()
    
def main():

    print("\n--- LOADING PDF ---")
    pdf_bytes = load_pdf_bytes(PDF_PATH)

    print("\n--- EXTRACTING ELEMENTS ---")
    elements = multi_modal_ingest(pdf_bytes)

    print(f"TOTAL ELEMENTS: {len(elements)}")

    chunks = []

    for el in elements:
        ch = normalize_element(el, pdf_bytes)

        if ch:
            chunks.append(ch)

    print(f"VALID CHUNKS: {len(chunks)}")

    print("\n--- BUILDING VECTOR STORE ---")
    build_faiss(chunks, "vector_store_mm")

    print("\nDONE\n")

if __name__ == "__main__":
    main()
