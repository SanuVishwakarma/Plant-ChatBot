from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security.api_key import APIKeyHeader, APIKey
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from typing import List, Optional
from datetime import datetime
import asyncio

from ragbase.chain import create_chain, ask_question
from ragbase.config import Config
from ragbase.ingestor import JsonIngestor
from ragbase.model import create_llm
from ragbase.retriever import create_retriever

app = FastAPI(title="Plant Shop Assistant API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key configuration
API_KEY_NAME = "Plant-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Store API keys (in production, use a database)
API_KEYS = {
    os.getenv("API_KEY"): {
        "client": "default",
        "created_at": datetime.now()
    }
}

class QuestionRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    session_id: str

async def get_api_key(api_key_header: str = Security(api_key_header)) -> str:
    if api_key_header in API_KEYS:
        return api_key_header
    raise HTTPException(
        status_code=403,
        detail="Invalid or missing API Key"
    )

# Initialize QA chain
def initialize_qa_chain():
    try:
        json_path = Config.Path.DATA_DIR / "shopyournursery.plants.json"
        vector_store = JsonIngestor().ingest(json_path)
        llm = create_llm()
        retriever = create_retriever(llm, vector_store=vector_store)
        return create_chain(llm, retriever)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize QA chain: {str(e)}"
        )

qa_chain = initialize_qa_chain()

@app.get("/")
async def root():
    return {"status": "online", "version": "1.0.0"}

@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: QuestionRequest,
    api_key: APIKey = Depends(get_api_key)
):
    try:
        session_id = request.session_id or f"session-{datetime.now().timestamp()}"
        response_chunks = []
        
        async for event in ask_question(qa_chain, request.question, session_id):
            if isinstance(event, str):
                response_chunks.append(event)
        
        full_response = "".join(response_chunks)
        
        return ChatResponse(
            answer=full_response,
            session_id=session_id
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )

@app.get("/health")
async def health_check(api_key: APIKey = Depends(get_api_key)):
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }