import os
import requests
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
import google.generativeai as genai
from PyPDF2 import PdfReader
from io import BytesIO

# Load environment variables (GOOGLE_API_KEY)
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# FastAPI instance
app = FastAPI()

# Input schema
class RunRequest(BaseModel):
    documents: str  # PDF URL
    questions: List[str]

# Output schema
class RunResponse(BaseModel):
    answers: List[str]

# Helper to extract text from a PDF URL
def extract_text_from_pdf(url: str) -> str:
    try:
        response = requests.get(url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to fetch PDF from URL")

        pdf_reader = PdfReader(BytesIO(response.content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint
@app.post("/hackrx/run", response_model=RunResponse)
async def run_query(request: RunRequest):
    # Extract PDF text
    context_text = extract_text_from_pdf(request.documents)

    # Use Gemini to answer each question
    model = genai.GenerativeModel("gemini-2.5-flash")  # or your preferred model
    answers = []

    for question in request.questions:
        prompt = f"""Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"""
        try:
            response = model.generate_content(prompt)
            answer = response.text.strip()
        except Exception as e:
            answer = f"Error generating answer: {str(e)}"

        answers.append(answer)

    return RunResponse(answers=answers)
