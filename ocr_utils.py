from pdf2image import convert_from_bytes
import pytesseract
import re

# ------------------------------------------------------------------------------
# PDF → convert page to image → OCR image → extract table text → clean → format
# ------------------------------------------------------------------------------

# Tesseract EXE path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"     

def extract_to_pages(pdf_bytes, page_number):

    '''
    Extract a specific page from a PDF as an image.
    
    Args:
        pdf_bytes (bytes): The PDF file in bytes.
        page_number (int): The page number to extract (1-indexed).

    Returns:
        PIL.Image: The extracted page as an image.
    '''

    pages = convert_from_bytes(pdf_bytes, dpi=300)    
    return pages[page_number - 1]                     


def ocr_table_to_markdown(ocr_text):
    '''
    Convert OCR-extracted table text to Markdown format.

    Args:
        ocr_text (str): The OCR-extracted text containing the table.

    Returns:
        str: The table formatted in Markdown.
    '''
    # Clean the OCR text
    lines = [line.strip() for line in ocr_text.split('\n') if line.strip()]     # ["Layer Type    Complexity"]

    # Split columns using multiple spaces
    rows = []
    for line in lines:
        cols = re.split(r"\s{2,}", line)        # ["Layer Type", "Complexity"]
        rows.append(cols)


    markdown_rows = []
    for cols in rows:
        markdown_rows.append('|' + '|'.join(cols) + '|')    # | Layer Type | Complexity |

    return "\n".join(markdown_rows)
