import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to path so we can import orchestrator and agents directly
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from orchestrator import OrchestratorAgent

app = FastAPI(
    title="VoyageAI - Autonomous Tourism Recommendation Engine",
    description="Multi-agent tourism recommendation API using ChromaDB, LlamaIndex, and GGUF LLMs",
    version="1.0.0"
)

# Enable CORS for local dev servers (Vite: 5173, Next.js: 3000, 3001)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy singleton orchestrator initialization
orchestrator_instance: Optional[OrchestratorAgent] = None

def get_orchestrator() -> OrchestratorAgent:
    global orchestrator_instance
    if orchestrator_instance is None:
        print("[FastAPI] Initializing Orchestrator Agent singleton...", flush=True)
        orchestrator_instance = OrchestratorAgent()
        print("[FastAPI] Orchestrator Agent ready!", flush=True)
    return orchestrator_instance

# Pydantic schemas
class ChatRequest(BaseModel):
    message: str
    target_count: Optional[int] = 10

class DestinationPlace(BaseModel):
    name: str
    text_content: Optional[str] = ""
    latitude: Optional[float] = 0.0
    longitude: Optional[float] = 0.0
    place_of_interests: Optional[List[str]] = []
    vibe: Optional[List[str]] = []
    place_of_activites: Optional[List[str]] = []
    country: Optional[str] = "International"
    state: Optional[str] = ""
    locality: Optional[Any] = []
    image_url: Optional[str] = ""
    confidence: Optional[str] = ""
    confidence_score: Optional[float] = 0.0
    evaluation_reason: Optional[str] = ""
    source_url: Optional[str] = ""
    source: Optional[str] = "Hybrid"
    raw_extract: Optional[str] = ""

class ChatResponse(BaseModel):
    success: bool
    user_query: str
    preferences: Dict[str, Any]
    places: List[DestinationPlace]
    total_places: int
    source: str
    formatted_text: str

class ConversationChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    target_count: Optional[int] = 6

class ConversationChatResponse(BaseModel):
    session_id: str
    success: bool
    user_query: str
    assistant_message: str
    places: List[DestinationPlace]
    total_places: int
    cumulative_preferences: Dict[str, Any]
    is_dissatisfied: bool
    strict_conjunction: bool
    history_turns: int

@app.on_event("startup")
async def startup_event():
    # Warm up orchestrator on start
    get_orchestrator()

@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "VoyageAI Multi-Agent Tourism Recommendation API",
        "endpoints": {
            "chat": "POST /api/chat",
            "conversation_chat": "POST /api/chat/conversation",
            "session_reset": "POST /api/chat/session/reset",
            "sample_places": "GET /api/places",
            "health": "GET /api/health"
        }
    }

@app.get("/api/health")
def health_check():
    return {"status": "ok", "agents_loaded": orchestrator_instance is not None}

@app.post("/api/chat", response_model=ChatResponse)
def chat_with_agent(req: ChatRequest):
    query = req.message.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query message cannot be empty.")
    
    agent = get_orchestrator()
    try:
        structured_result = agent.process_request_structured(
            user_input=query,
            target_count=req.target_count or 10
        )
        return structured_result
    except Exception as e:
        print(f"[FastAPI] Error processing request: {e}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/conversation", response_model=ConversationChatResponse)
def conversation_turn(req: ConversationChatRequest):
    query = req.message.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query message cannot be empty.")
    
    agent = get_orchestrator()
    try:
        result = agent.process_conversational_turn(
            user_input=query,
            session_id=req.session_id,
            target_count=req.target_count or 6
        )
        return result
    except Exception as e:
        print(f"[FastAPI] Error in conversational turn: {e}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/session/reset")
def reset_conversation_session(session_id: str):
    from backend.session_manager import session_manager
    success = session_manager.reset_session(session_id)
    return {"success": success, "session_id": session_id, "message": "Session reset successfully."}

class SummarizeDestinationRequest(BaseModel):
    name: str
    raw_extract: Optional[str] = ""
    user_context: Optional[str] = ""

class SummarizeDestinationResponse(BaseModel):
    name: str
    summary_500_words: str
    country: Optional[str] = ""
    state: Optional[str] = ""
    locality: Optional[Any] = []
    vibe: Optional[List[str]] = []
    place_of_activites: Optional[List[str]] = []
    is_ai_generated: bool = True

@app.post("/api/destination/summarize", response_model=SummarizeDestinationResponse)
def summarize_destination_on_demand(req: SummarizeDestinationRequest):
    """
    On-Demand Summarization (Solution 2):
    Generates a rich 500-word AI travel guide ONLY when requested for this specific destination,
    and updates ChromaDB vector store for future queries.
    """
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Destination name is required.")

    agent = get_orchestrator()
    try:
        summary_data = agent.search_agent.generate_ondemand_summary(
            title=name,
            raw_extract=req.raw_extract or "",
            user_context=req.user_context or ""
        )
        # Update ChromaDB / local sample with the new rich 500-word summary
        agent.db_agent.insert_places([{
            "name": name,
            "text_content": summary_data["summary_500_words"],
            "country": summary_data.get("country", "International"),
            "state": summary_data.get("state", ""),
            "locality": summary_data.get("locality", []),
            "vibe": summary_data.get("vibe", []),
            "place_of_activites": summary_data.get("place_of_activites", [])
        }])
        return summary_data
    except Exception as e:
        print(f"[FastAPI] Error in on-demand summarization: {e}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/places")
def get_cached_places():
    """Returns currently indexed places in Chroma DB / sample file."""
    import json
    sample_path = ROOT_DIR / "chroma_sample.json"
    if sample_path.exists():
        try:
            with open(sample_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {"count": len(data), "places": data}
        except Exception as e:
            return {"error": str(e), "places": []}
    return {"count": 0, "places": []}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
