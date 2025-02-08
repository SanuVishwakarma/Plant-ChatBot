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

# Configure CORS - adjusted for Hugging Face
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key configuration - modified for Hugging Face
API_KEY_NAME = "Plant-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

class QuestionRequest(BaseModel):
    question: str
    

class ChatResponse(BaseModel):
    answer: str
    

# Modified API key handling for Hugging Face
async def get_api_key(api_key_header: str = Security(api_key_header)) -> str:
    if Config.DEBUG:  # Allow no API key in debug mode
        return api_key_header or "debug"
    if api_key_header in API_KEYS:
        return api_key_header
    raise HTTPException(
        status_code=403,
        detail="Invalid or missing API Key"
    )

# Initialize QA chain with error handling
def initialize_qa_chain():
    try:
        # Ensure data directory exists
        os.makedirs(os.path.join(os.getcwd(), "data"), exist_ok=True)
        os.makedirs(os.path.join(os.getcwd(), "database"), exist_ok=True)
        
        json_path = Config.Path.DATA_DIR / "shopyournursery.plants.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Data file not found at {json_path}")
            
        vector_store = JsonIngestor().ingest(json_path)
        llm = create_llm()
        retriever = create_retriever(llm, vector_store=vector_store)
        return create_chain(llm, retriever)
    except Exception as e:
        print(f"Initialization error: {str(e)}")  # Add logging
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize QA chain: {str(e)}"
        )

# Global variables
qa_chain = None
API_KEYS = {os.getenv("API_KEY", "default-key"): {"client": "default", "created_at": datetime.now()}}

@app.on_event("startup")
async def startup_event():
    global qa_chain
    qa_chain = initialize_qa_chain()

@app.get("/")
async def root():
    return {
        "status": "online",
        "version": Config.API_VERSION,
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: QuestionRequest,
    api_key: APIKey = Depends(get_api_key)
):
    try:
        if qa_chain is None:
            raise HTTPException(
                status_code=503,
                detail="Service is initializing. Please try again in a few moments."
            )
            
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
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "debug_mode": Config.DEBUG,
        "model": Config.Model.REMOTE_LLM if not Config.Model.USE_LOCAL else Config.Model.LOCAL_LLM
    }