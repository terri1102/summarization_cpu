from fastapi import FastAPI, status
from typing import Dict

from app.model.inference import TextSummarizer

app = FastAPI()
summarizer = TextSummarizer()

@app.get("/health", tags=["healthcheck"], status_code=status.HTTP_200_OK)
def health() -> Dict[str, str]:
    return {"status": "OK"}

@app.post("/summarize", tags=["summarizer"], status_code=status.HTTP_200_OK)
def summarize(data: Dict[str, str]) -> Dict[str, str]:
    text = data.get("text")
    if not text:
        return {"error": "text is required"}
    result = summarizer.summarize(text)
    return {"summary": result}