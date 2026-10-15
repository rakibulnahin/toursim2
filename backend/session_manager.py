import time
import uuid
from typing import Dict, List, Set, Any, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: str  # "user" | "assistant"
    content: str
    places: Optional[List[Dict[str, Any]]] = []
    timestamp: float = Field(default_factory=time.time)


class SessionState:
    def __init__(self, session_id: str):
        self.session_id: str = session_id
        self.created_at: float = time.time()
        self.last_active: float = time.time()
        self.messages: List[ChatMessage] = []
        self.cumulative_preferences: Dict[str, Any] = {
            "duration": "unspecified",
            "place_of_interests": [],
            "vibe": [],
            "place_of_activites": [],
            "specific_named_places": [],
            "country_names": [],
            "city_names": [],
            "raw_user_input": ""
        }
        self.seen_place_names: Set[str] = set()
        self.strict_conjunction: bool = False
        self.dissatisfaction_count: int = 0

    def touch(self):
        self.last_active = time.time()

    def add_message(self, role: str, content: str, places: Optional[List[Dict[str, Any]]] = None) -> ChatMessage:
        self.touch()
        msg = ChatMessage(role=role, content=content, places=places or [])
        self.messages.append(msg)
        if places:
            for p in places:
                name = str(p.get("name", "")).strip().lower()
                if name:
                    self.seen_place_names.add(name)
        return msg

    def get_recent_dialogue(self, max_turns: int = 4) -> List[Dict[str, str]]:
        recent = self.messages[-max_turns * 2:]
        return [{"role": m.role, "content": m.content} for m in recent]

    def reset(self):
        self.messages = []
        self.cumulative_preferences = {
            "duration": "unspecified",
            "place_of_interests": [],
            "vibe": [],
            "place_of_activites": [],
            "specific_named_places": [],
            "country_names": [],
            "city_names": [],
            "raw_user_input": ""
        }
        self.seen_place_names = set()
        self.strict_conjunction = False
        self.dissatisfaction_count = 0
        self.touch()


class SessionManager:
    """
    In-memory session manager with TTL auto-eviction.
    """
    def __init__(self, ttl_seconds: int = 86400):  # 24 hour TTL
        self.sessions: Dict[str, SessionState] = {}
        self.ttl_seconds: int = ttl_seconds

    def get_or_create_session(self, session_id: Optional[str] = None) -> SessionState:
        self._cleanup_expired()
        if not session_id or session_id not in self.sessions:
            new_id = session_id or str(uuid.uuid4())
            self.sessions[new_id] = SessionState(new_id)
            return self.sessions[new_id]
        
        session = self.sessions[session_id]
        session.touch()
        return session

    def reset_session(self, session_id: str) -> bool:
        if session_id in self.sessions:
            self.sessions[session_id].reset()
            return True
        return False

    def _cleanup_expired(self):
        now = time.time()
        expired_keys = [
            sid for sid, s in self.sessions.items()
            if now - s.last_active > self.ttl_seconds
        ]
        for sid in expired_keys:
            del self.sessions[sid]


# Global session manager singleton
session_manager = SessionManager()
