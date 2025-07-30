import fitz  
import requests
from tempfile import NamedTemporaryFile

def download_and_extract_text(pdf_url: str) -> str:
    response = requests.get(pdf_url)
    response.raise_for_status()

    with NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(response.content)
        tmp.flush()

        doc = fitz.open(tmp.name)
        full_text = ""
        for page in doc:
            full_text += page.get_text()

    return full_text
