import fitz  # PyMuPDF
import httpx
import tempfile

def download_and_extract_text(pdf_url: str) -> str:
    response = httpx.get(pdf_url)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
        tmp_file.write(response.content)
        tmp_path = tmp_file.name

    doc = fitz.open(tmp_path)
    text = ""
    for page in doc:
        text += page.get_text()

    return text
