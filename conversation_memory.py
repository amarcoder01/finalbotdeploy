"""
Conversation Memory Service - Maintains user session context and history
Provides context-aware conversations for better user experience
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from logger import BotLogger
from sqlalchemy.future import select
from db import AsyncSessionLocal
from models import User, UserPreference
import asyncio

logger = BotLogger(__name__)

class ConversationMemory:
    """Service for managing user conversation history and context"""
    
    def __init__(self, session_timeout: int = 3600):  # 1 hour timeout
        """
        Initialize conversation memory service
        
        Args:
            session_timeout (int): Session timeout in seconds (default: 1 hour)
        """
        self.sessions: Dict[int, Dict] = {}
        self.session_timeout = session_timeout
        logger.info("Conversation memory service initialized")
    
    def add_message(self, user_id: int, message: str, response: str, message_type: str = "text") -> None:
        """
        Add a message and response to user's conversation history
        
        Args:
            user_id (int): Telegram user ID
            message (str): User's message
            response (str): Bot's response
            message_type (str): Type of message (text, command, etc.)
        """
        try:
            current_time = datetime.utcnow()
            
            # Initialize session if not exists
            if user_id not in self.sessions:
                self.sessions[user_id] = {
                    'messages': [],
                    'user_preferences': {},
                    'last_activity': current_time,
                    'session_start': current_time
                }
            
            # Clean old sessions periodically
            self._cleanup_old_sessions()
            
            # Add new message to history
            self.sessions[user_id]['messages'].append({
                'timestamp': current_time.isoformat(),
                'user_message': message[:500],  # Limit message length for memory
                'bot_response': response[:500],  # Limit response length for memory
                'message_type': message_type
            })
            
            # Update last activity
            self.sessions[user_id]['last_activity'] = current_time
            
            # Keep only last 20 messages to prevent memory bloat
            if len(self.sessions[user_id]['messages']) > 20:
                self.sessions[user_id]['messages'] = self.sessions[user_id]['messages'][-20:]
            
            logger.debug(f"Added message to history for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error adding message to history: {e}")
    
    def get_conversation_context(self, user_id: int, include_last_n: int = 5) -> str:
        """
        Get recent conversation context for AI processing
        
        Args:
            user_id (int): Telegram user ID
            include_last_n (int): Number of recent messages to include
            
        Returns:
            str: Formatted conversation context
        """
        try:
            if user_id not in self.sessions:
                return ""
            
            messages = self.sessions[user_id]['messages']
            if not messages:
                return ""
            
            # Get last N messages
            recent_messages = messages[-include_last_n:] if len(messages) >= include_last_n else messages
            
            context_parts = []
            for msg in recent_messages:
                # Format: "User: message | Bot: response"
                context_parts.append(
                    f"User: {msg['user_message'][:100]}... | "
                    f"Bot: {msg['bot_response'][:100]}..."
                )
            
            context = "Recent conversation:\n" + "\n".join(context_parts)
            logger.debug(f"Generated context for user {user_id}: {len(context)} chars")
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting conversation context: {e}")
            return ""
    
    async def get_user_preferences_db(self, user_id: int) -> Dict:
        """Get user preferences from the database"""
        try:
            async with AsyncSessionLocal() as session:
                user = await session.execute(select(User).where(User.telegram_id == str(user_id)))
                user_obj = user.scalars().first()
                if not user_obj:
                    return {}
                prefs = await session.execute(select(UserPreference).where(UserPreference.user_id == user_obj.id))
                prefs_objs = prefs.scalars().all()
                return {pref.key: pref.value for pref in prefs_objs}
        except Exception as e:
            logger.error(f"Error fetching user preferences from DB: {e}")
            return {}

    async def update_user_preference_db(self, user_id: int, key: str, value: str) -> None:
        """Update or create a user preference in the database"""
        try:
            async with AsyncSessionLocal() as session:
                user = await session.execute(select(User).where(User.telegram_id == str(user_id)))
                user_obj = user.scalars().first()
                if not user_obj:
                    user_obj = User(telegram_id=str(user_id))
                    session.add(user_obj)
                    await session.commit()
                    await session.refresh(user_obj)
                pref = await session.execute(select(UserPreference).where(UserPreference.user_id == user_obj.id, UserPreference.key == key))
                pref_obj = pref.scalars().first()
                if pref_obj:
                    pref_obj.value = value
                else:
                    pref_obj = UserPreference(user_id=user_obj.id, key=key, value=value)
                    session.add(pref_obj)
                await session.commit()
                logger.info(f"Updated DB preference for user {user_id}: {key} = {value}")
        except Exception as e:
            logger.error(f"Error updating user preference in DB: {e}")
    
    def get_session_stats(self, user_id: int) -> Dict:
        """
        Get session statistics for a user
        
        Args:
            user_id (int): Telegram user ID
            
        Returns:
            Dict: Session statistics
        """
        try:
            if user_id not in self.sessions:
                return {}
            
            session = self.sessions[user_id]
            current_time = datetime.utcnow()
            
            return {
                'message_count': len(session['messages']),
                'session_duration': str(current_time - session['session_start']),
                'last_activity': session['last_activity'].strftime('%Y-%m-%d %H:%M:%S'),
                'preferences_count': len(session.get('user_preferences', {}))
            }
            
        except Exception as e:
            logger.error(f"Error getting session stats: {e}")
            return {}
    
    def _cleanup_old_sessions(self) -> None:
        """Clean up sessions that have exceeded the timeout"""
        try:
            current_time = datetime.utcnow()
            timeout_threshold = current_time - timedelta(seconds=self.session_timeout)
            
            expired_users = []
            for user_id, session in self.sessions.items():
                if session['last_activity'] < timeout_threshold:
                    expired_users.append(user_id)
            
            for user_id in expired_users:
                del self.sessions[user_id]
                logger.info(f"Cleaned up expired session for user {user_id}")
                
        except Exception as e:
            logger.error(f"Error cleaning up sessions: {e}")
    
    def clear_user_session(self, user_id: int) -> bool:
        """
        Clear a user's session data
        
        Args:
            user_id (int): Telegram user ID
            
        Returns:
            bool: True if session was cleared, False if not found
        """
        try:
            if user_id in self.sessions:
                del self.sessions[user_id]
                logger.info(f"Cleared session for user {user_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error clearing user session: {e}")
            return False
    
    def get_active_sessions_count(self) -> int:
        """
        Get count of active sessions
        
        Returns:
            int: Number of active sessions
        """
        try:
            self._cleanup_old_sessions()
            return len(self.sessions)
            
        except Exception as e:
            logger.error(f"Error getting active sessions count: {e}")
            return 0