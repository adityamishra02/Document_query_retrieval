from document_reader import load_document_from_url

def test_loader():
    url = "https://example.com/sample.pdf"
    chunks = load_document_from_url(url)
    assert len(chunks) > 0
    print("Test passed!")

if __name__ == "__main__":
    test_loader()
