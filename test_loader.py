from document_loader import download_and_extract_text

pdf_url = "https://hackrx.in/policies/EDLHLGA23009V012223.pdf"
text = download_and_extract_text(pdf_url)

# Just print the beginning to verify
print(text[:1000])
