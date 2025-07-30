from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import List
from main import run_pipeline

app = FastAPI()

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

@app.post("/hackrx/run")
async def handle_query(request: QueryRequest):
    try:
        answers = run_pipeline(request.documents, request.questions)
        return {"answers": answers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
