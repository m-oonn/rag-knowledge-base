from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from src.services.rag_service import get_rag_service

router = APIRouter()
rag_service = get_rag_service()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class QueryResponse(BaseModel):
    query: str
    answer: str
    source_documents: List[Dict[str, Any]]

@router.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    try:
        result = rag_service.query(request.query, request.top_k)
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
