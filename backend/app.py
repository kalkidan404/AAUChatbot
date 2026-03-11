from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline import ask

app = FastAPI()

class Question(BaseModel):
    question: str


@app.post("/ask")
def ask_question(q: Question):
    # Updated to get chunks as well
    answer, sources, chunks = ask(q.question)

    return {
        "answer": answer,
        "sources": sources,
        "chunks": chunks
    }