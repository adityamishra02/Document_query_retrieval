import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from io import BytesIO
import google.generativeai as genai

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# FastAPI instance
app = FastAPI()

# Pydantic models for request and response
class RunRequest(BaseModel):
    documents: str  # URL to the PDF file
    questions: List[str]

class RunResponse(BaseModel):
    answers: List[str]

# Helper function to extract text from a PDF URL
def extract_text_from_pdf(url: str) -> str:
    try:
        response = requests.get(url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to fetch PDF from URL.")

        pdf_reader = PdfReader(BytesIO(response.content))
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading PDF: {str(e)}")

# Main endpoint
@app.post("/hackrx/run", response_model=RunResponse)
async def run_hackrx(request: RunRequest):
    # Extract text from the given PDF URL
    context = extract_text_from_pdf(request.documents)

    # Initialize Gemini model
    model = genai.GenerativeModel("gemini-2.5-flash")

    answers = []
    for question in request.questions:
        prompt = f"""
You are an assistant that answers questions about documents.

ONLY use the information below to answer the question. Be factual, direct, and include all relevant details from the document. Do not explain your reasoning. Do not use any external knowledge.

Below are examples of how you should answer:

Example 1
Q: What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?
A: A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.

Example 2
Q: What is the waiting period for pre-existing diseases (PED) to be covered?
A: There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered.

Example 3
Q: Does this policy cover maternity expenses, and what are the conditions?
A: Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months. The benefit is limited to two deliveries or terminations during the policy period.

Reference Document:
\"\"\"
{context}
\"\"\"

Q: {question}
A:"""
        try:
            response = model.generate_content(prompt)
            answers.append(response.text.strip())
        except Exception as e:
            answers.append(f"Error: {str(e)}")

    return RunResponse(answers=answers)
