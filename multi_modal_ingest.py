from ocr_utils import extract_to_pages, ocr_table_to_markdown

from unstructured.partition.pdf import partition_pdf
from pdf2image import convert_from_bytes
import pytesseract
from bs4 import BeautifulSoup
import tempfile


# ---------------------------------------------------------------------------------------------------
# PDF → Unstructured Elements → Normalize Elements → Multi-Modal Chunks → Embeddings → FAISS Retriver
# ---------------------------------------------------------------------------------------------------

def multi_modal_ingest(file_bytes: bytes):
    """
    Ingest a PDF using Unstructured.io and return a list of structured elements.
    Each element will be converted into a plain dict for easy storage & retrieval.

    Args:
        file_bytes (bytes): The PDF file in bytes.

    Returns:
        list: List of structured elements as dictionaries.
    """

    # Save the uploaded bytes as a temporary PDF file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name


    # Partition PDF - Unstructured will extract text, tables, images, OCR etc.
    elements = partition_pdf(
        filename=tmp_path,
        include_page_breaks=True,
        infer_table_structure=True,  
        strategy="hi_res",           
    )


    structured_output = []

    for el in elements:
        structured_output.append({
            "type": el.__class__.__name__,            # e.g., NarrativeText, Table, Title, Picture
            "content": el.text if hasattr(el, "text") else "",
            "page": el.metadata.page_number,
            "category": el.metadata.category_depth,  # <=
            "filetype": el.metadata.filetype,
            "image_path": getattr(el, "image_path", None),
        })

    return structured_output


def normalize_element(el, pdf_bytes):
    """
    Convert the Unstructured element into a unified RAG chunk.
    
    Args:
        el (dict): The structured element dictionary.
        pdf_bytes (bytes): The original PDF file in bytes.

    Returns:
        dict: A normalized RAG chunk dictionary.
    """

    element_type = el['type']
    page = el.get('page')
    text = el.get("content", "").strip()
    html = el.get("metadata", {}).get("text_as_html")
    img_path = el.get("image_path")

    # -------------------------------------
    # CASE 1: NORMAL TEXT ELEMENTS
    # -------------------------------------
    if element_type not in ["Table", "Image", "Picture"]:
        return {
            "type": "text",
            "page": page,
            "content": text,
            "image_path": None,
            "embedding_text": f"Text: {text}"
        }

    # -------------------------------------
    # CASE 2: TABLE HANDLING
    # -------------------------------------
    if element_type == "Table":

        
        if html:
            soup = BeautifulSoup(html, "html.parser")
            rows = soup.find_all("tr")

            markdown = []
            for r in rows:
                cols = [c.get_text(strip=True) for c in r.find_all(["td", "th"])]

                
                if len(cols) > 1:
                    markdown.append("| " + " | ".join(cols) + " |")

            
            if markdown:
                table_md = "\n".join(markdown)

                return {
                    "type": "table",
                    "page": page,
                    "content": table_md,
                    "image_path": None,
                    "embedding_text": f"Table: {table_md}",
                }

        
        if img_path:
            ocr_text = pytesseract.image_to_string(img_path).strip()

            return {
                "type": "ocr",
                "page": page,
                "content": ocr_text,
                "image_path": img_path,
                "embedding_text": f"OCR: {ocr_text}",
            }

        # 2C: fallback → treat as text
        return {
            "type": "text",
            "page": page,
            "content": text,
            "image_path": None,
            "embedding_text": f"Text: {text}",
        }

    # -------------------------------------
    # CASE 3: IMAGE / PICTURE → OCR
    # -------------------------------------
    if element_type in ["Image", "Picture"] and img_path:
        ocr_text = pytesseract.image_to_string(img_path).strip()

        return {
            "type": "ocr",
            "page": page,
            "content": ocr_text,
            "image_path": img_path,
            "embedding_text": f"OCR: {ocr_text}",
        }

    # -------------------------------------
    # DEFAULT FALLBACK
    # -------------------------------------
    return None





