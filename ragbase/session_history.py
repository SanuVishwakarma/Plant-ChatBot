from datetime import datetime, timedelta
from typing import Dict, Optional
from langchain_community.chat_message_histories import ChatMessageHistory

class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        self.session_timeout = timedelta(hours=1)

    def get_session_history(self, session_id: str) -> ChatMessageHistory:
        current_time = datetime.now()
        
        # Clean expired sessions
        self._clean_expired_sessions()
        
        # Create or update session
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'history': ChatMessageHistory(),
                'created_at': current_time,
                'last_accessed': current_time
            }
        else:
            self.sessions[session_id]['last_accessed'] = current_time
            
        return self.sessions[session_id]['history']

    def _clean_expired_sessions(self):
        current_time = datetime.now()
        expired_sessions = [
            session_id for session_id, session_data in self.sessions.items()
            if current_time - session_data['last_accessed'] > self.session_timeout
        ]
        for session_id in expired_sessions:
            del self.sessions[session_id]

    def get_session_info(self, session_id: str) -> Optional[Dict]:
        if session_id in self.sessions:
            return {
                'created_at': self.sessions[session_id]['created_at'],
                'last_accessed': self.sessions[session_id]['last_accessed'],
                'message_count': len(self.sessions[session_id]['history'].messages)
            }
        return None

# Create a global session manager instance
session_manager = SessionManager()

def get_session_history(session_id: str) -> ChatMessageHistory:
    return session_manager.get_session_history(session_id)