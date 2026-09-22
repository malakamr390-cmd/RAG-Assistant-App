from fastapi import APIRouter

from backend.app.schemas.query import QueryRequest, QueryResponse
from backend.app.services.retrieval import retrieve
from backend.app.services.generation import generate_answer

router = APIRouter()

@router.get("/health")
def health():
    return {"status": "ok"}

@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    retrieved_chunks = retrieve(request.question)
    answer = generate_answer(request.question, retrieved_chunks)
    sources = list({f"{c['source']} p.{c['page']}" for c in retrieved_chunks})
    return QueryResponse(answer=answer, sources=sources)