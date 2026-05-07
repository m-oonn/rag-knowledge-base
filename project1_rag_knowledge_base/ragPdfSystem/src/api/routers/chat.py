from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import List, Optional, Any
from pydantic import BaseModel
import uuid

from src.database.sql_session import get_db
from src.database.models import ChatSession, ChatInteraction, User, KnowledgeBase, Assistant
from src.services.rag_service import get_rag_service
from src.services.memory_service import MemorySystem
from src.api.dependencies import get_current_user

router = APIRouter()
rag_service = get_rag_service()
memory_system = MemorySystem()

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    kb_id: Optional[int] = None # Deprecated/Override
    assistant_id: Optional[int] = None # New: Select Assistant
    top_k: int = 5

class ChatResponse(BaseModel):
    session_id: str
    query: str
    answer: str
    source_documents: List[Any]

class SessionOut(BaseModel):
    id: int
    session_uid: str
    title: Optional[str]
    created_at: Any
    assistant_id: Optional[int]
    
    class Config:
        from_attributes = True

class MessageOut(BaseModel):
    query: str
    answer: str
    created_at: Any
    
    class Config:
        from_attributes = True

@router.post("/", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # 0. Resolve Assistant & Configuration
    assistant = None
    kb_ids = []
    
    if request.assistant_id:
        assistant = db.query(Assistant).filter(Assistant.id == request.assistant_id).first()
        if not assistant:
             raise HTTPException(status_code=404, detail="Assistant not found")
        # Optional: check ownership or public
        if assistant.user_id != current_user.id:
             raise HTTPException(status_code=403, detail="Not authorized for this assistant")
        
        # Load config from assistant
        kb_ids = assistant.kb_ids or []
    elif request.kb_id:
        # Legacy/Direct KB mode
        kb_ids = [request.kb_id]
    
    # 1. Validate KBs (if any)
    valid_kb_ids = []
    if kb_ids:
        kbs = db.query(KnowledgeBase).filter(KnowledgeBase.id.in_(kb_ids)).all()
        for kb in kbs:
            if kb.owner_id == current_user.id or kb.is_public:
                valid_kb_ids.append(kb.id)
    
    # 2. Manage Session
    session_uid = request.session_id
    if not session_uid:
        session_uid = str(uuid.uuid4())
        chat_session = ChatSession(
            session_uid=session_uid,
            user_id=current_user.id,
            assistant_id=request.assistant_id,
            title=request.query[:50]  # Simple title generation
        )
        db.add(chat_session)
        db.commit()
        db.refresh(chat_session)
    else:
        chat_session = db.query(ChatSession).filter(ChatSession.session_uid == session_uid).first()
        if not chat_session:
            chat_session = ChatSession(
                session_uid=session_uid,
                user_id=current_user.id,
                assistant_id=request.assistant_id,
                title=request.query[:50]
            )
            db.add(chat_session)
            db.commit()
            db.refresh(chat_session)
        
        if chat_session.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized to access this session")

    # 3. Call RAG Service
    # Pass assistant config if available
    assistant_config = {
        "llm_model": assistant.llm_model if assistant else "qwen-max",
        "temperature": assistant.temperature if assistant else 0.7,
        "system_prompt": assistant.system_prompt if assistant else None,
        "memory_config": assistant.memory_config if assistant else None,
        "rag_config": assistant.rag_config if assistant else None,
        "tool_config": assistant.tool_config if assistant else None,
        "agent_ids": assistant.agent_ids if assistant else None
    }
    
    result = rag_service.query(
        query_text=request.query,
        top_k=request.top_k,
        session_id=session_uid,
        kb_ids=valid_kb_ids, # Pass list of IDs
        assistant_config=assistant_config
    )
    
    # 4. Save Interaction
    interaction = ChatInteraction(
        session_id=chat_session.id,
        kb_id=valid_kb_ids[0] if valid_kb_ids else None, # Store primary KB or None
        query=request.query,
        answer=result["answer"],
        retrieved_docs=result.get("source_documents"), # JSON field
        metrics={} # Placeholder for metrics
    )
    db.add(interaction)
    db.commit()
    
    return ChatResponse(
        session_id=session_uid,
        query=request.query,
        answer=result["answer"],
        source_documents=result.get("source_documents", [])
    )

@router.get("/sessions", response_model=List[SessionOut])
def list_sessions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return db.query(ChatSession).filter(ChatSession.user_id == current_user.id).order_by(ChatSession.created_at.desc()).all()

@router.get("/sessions/{session_id}/messages", response_model=List[MessageOut])
def get_session_messages(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    chat_session = db.query(ChatSession).filter(ChatSession.session_uid == session_id).first()
    if not chat_session:
        raise HTTPException(status_code=404, detail="Session not found")
    if chat_session.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
        
    return db.query(ChatInteraction).filter(ChatInteraction.session_id == chat_session.id).order_by(ChatInteraction.created_at.asc()).all()

@router.delete("/sessions/{session_id}")
def delete_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    chat_session = db.query(ChatSession).filter(ChatSession.session_uid == session_id).first()
    if not chat_session:
        raise HTTPException(status_code=404, detail="Session not found")
    if chat_session.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # Delete interactions first
    db.query(ChatInteraction).filter(ChatInteraction.session_id == chat_session.id).delete()
    db.delete(chat_session)
    db.commit()
    
    # Clear short term memory from Redis
    try:
        memory_system.clear_short_term_memory(session_id)
    except Exception as e:
        print(f"Error clearing memory: {e}")
    
    return {"message": "Session deleted"}

@router.delete("/sessions")
def batch_delete_sessions(
    session_ids: List[str], # List of session_uids
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    sessions = db.query(ChatSession).filter(ChatSession.session_uid.in_(session_ids)).all()
    deleted_count = 0
    
    for session in sessions:
        if session.user_id == current_user.id:
            db.query(ChatInteraction).filter(ChatInteraction.session_id == session.id).delete()
            db.delete(session)
            deleted_count += 1
            
    db.commit()
    return {"message": f"Deleted {deleted_count} sessions"}


@router.post("/stream")
def chat_stream(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """SSE streaming RAG chat endpoint."""
    # Resolve KBs (same logic as main chat endpoint)
    assistant = None
    kb_ids = []
    if request.assistant_id:
        assistant = db.query(Assistant).filter(Assistant.id == request.assistant_id).first()
        if assistant and (assistant.user_id == current_user.id):
            kb_ids = assistant.kb_ids or []
    elif request.kb_id:
        kb_ids = [request.kb_id]

    valid_kb_ids = []
    if kb_ids:
        kbs = db.query(KnowledgeBase).filter(KnowledgeBase.id.in_(kb_ids)).all()
        valid_kb_ids = [kb.id for kb in kbs if kb.owner_id == current_user.id or kb.is_public]

    # Session
    import uuid as _uuid
    session_uid = request.session_id or str(_uuid.uuid4())

    assistant_config = {
        "system_prompt": assistant.system_prompt if assistant else None,
        "memory_config": assistant.memory_config if assistant else None,
    } if assistant else None

    def generate():
        """SSE event generator."""
        # Build RAG prompt
        result = rag_service.query(
            query_text=request.query,
            top_k=request.top_k,
            session_id=session_uid,
            kb_ids=valid_kb_ids,
            assistant_config=assistant_config,
        )

        # Yield sources first
        sources_json = __import__("json").dumps([
            {"id": s.get("id", ""), "score": s.get("score", 0), "text": s.get("text", "")[:200]}
            for s in result.get("source_documents", [])
        ])
        yield f"event: sources\ndata: {sources_json}\n\n"

        # Stream the answer
        context = "\n\n".join(
            f"[{i+1}] {s.get('text', '')}"
            for i, s in enumerate(result.get("source_documents", []))
        )
        prompt = rag_service.llm_client.generate_response.__wrapped__ if hasattr(
            rag_service.llm_client.generate_response, "__wrapped__"
        ) else None

        # Use the LLM client's streaming directly
        rag_prompt = f"""基于以下上下文信息，回答问题。

上下文：
{context}

问题：{request.query}

要求：基于上下文回答，不添加外部知识。如无相关信息请说明。

回答："""

        for token in rag_service.llm_client.generate_stream(rag_prompt):
            yield f"data: {__import__('json').dumps({'token': token})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
