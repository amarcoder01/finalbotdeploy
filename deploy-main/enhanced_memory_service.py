"""Enhanced Memory Service - Integration layer for Intelligent Memory System
Provides backward compatibility while adding advanced memory capabilities
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import asdict

from intelligent_memory_system import (
    IntelligentMemorySystem, MemoryType, MemoryImportance, 
    MemoryEntry, intelligent_memory
)
from conversation_memory import ConversationMemory
from logger import BotLogger
from performance_cache import cache_result, with_connection_pool

logger = BotLogger(__name__)

class EnhancedMemoryService:
    """Enhanced memory service that combines traditional conversation memory
    with intelligent semantic memory capabilities
    """
    
    def __init__(self):
        """Initialize the enhanced memory service"""
        self.conversation_memory = ConversationMemory()
        self.intelligent_memory = intelligent_memory
        self.memory_integration_enabled = True
        
        logger.info("Enhanced Memory Service initialized")
    
    async def add_interaction(self, 
                            user_id: int, 
                            user_message: str, 
                            bot_response: str, 
                            message_type: str = "text",
                            context: Optional[Dict[str, Any]] = None,
                            importance: MemoryImportance = MemoryImportance.MEDIUM) -> str:
        """Add a user interaction to both conversation and intelligent memory
        
        Args:
            user_id: User ID
            user_message: User's message
            bot_response: Bot's response
            message_type: Type of message
            context: Additional context
            importance: Memory importance level
            
        Returns:
            Memory ID from intelligent memory system
        """
        try:
            # Add to traditional conversation memory for backward compatibility
            self.conversation_memory.add_message(
                user_id=user_id,
                message=user_message,
                response=bot_response,
                message_type=message_type
            )
            
            # Add to intelligent memory system
            memory_entry = MemoryEntry(
                user_id=user_id,
                memory_type=MemoryType.CONVERSATION,
                content=f"User: {user_message} | Bot: {bot_response}",
                context=f"Message type: {message_type}",
                importance=importance,
                tags=["conversation", message_type],
                metadata=context or {
                    "message_type": message_type,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            success = await self.intelligent_memory.add_memory(memory_entry)
            memory_id = "success" if success else "failed"
            
            # Extract and store preferences if detected
            await self._extract_preferences(user_id, user_message, bot_response)
            
            logger.debug(f"Added interaction to enhanced memory for user {user_id}")
            return memory_id
            
        except Exception as e:
            logger.error(f"Error adding interaction to enhanced memory: {e}")
            return ""
    
    async def get_contextual_response_data(self, 
                                         user_id: int, 
                                         current_query: str,
                                         include_conversation_context: bool = True) -> Dict[str, Any]:
        """Get comprehensive contextual data for generating responses
        
        Args:
            user_id: User ID
            current_query: Current user query
            include_conversation_context: Whether to include conversation context
            
        Returns:
            Dictionary with contextual data
        """
        try:
            response_data = {}
            
            # Get relevant memories using search
            relevant_memories = await self.intelligent_memory.search_memories(
                user_id=user_id,
                query=current_query,
                limit=10
            )
            
            # Get contextual memories
            contextual_memories = await self.intelligent_memory.get_contextual_memories(
                user_id=user_id,
                context=current_query,
                limit=5
            )
            response_data["contextual_memories"] = contextual_memories
            response_data["relevant_memories"] = [
                 {
                     "content": memory.get("content", ""),
                     "memory_type": memory.get("memory_type", ""),
                     "importance": memory.get("importance", ""),
                     "similarity": memory.get("similarity", 0.0),
                     "created_at": memory.get("created_at", datetime.utcnow()).isoformat() if hasattr(memory.get("created_at", datetime.utcnow()), 'isoformat') else str(memory.get("created_at", ""))
                 }
                 for memory in relevant_memories
             ]
            
            # Get traditional conversation context if requested
            if include_conversation_context:
                conversation_context = self.conversation_memory.get_conversation_context(
                    user_id=user_id,
                    include_last_n=5
                )
                response_data["conversation_context"] = conversation_context
            
            # Get user preferences
            user_preferences = await self._get_user_preferences(user_id)
            response_data["user_preferences"] = user_preferences
            
            # Get user patterns
            user_patterns = await self._get_user_patterns(user_id)
            response_data["user_patterns"] = user_patterns
            
            # Get memory statistics
            memory_stats = await self.intelligent_memory.get_memory_stats(user_id)
            response_data["memory_stats"] = memory_stats
            
            logger.debug(f"Generated contextual response data for user {user_id}")
            return response_data
            
        except Exception as e:
            logger.error(f"Error getting contextual response data: {e}")
            return {}
    
    async def add_trading_activity(self, 
                                 user_id: int, 
                                 symbol: str, 
                                 action: str, 
                                 quantity: float, 
                                 price: float,
                                 context: Optional[Dict[str, Any]] = None):
        """Record trading activity in memory
        
        Args:
            user_id: User ID
            symbol: Stock symbol
            action: Trading action (buy/sell)
            quantity: Quantity traded
            price: Price per share
            context: Additional context
        """
        try:
             content = f"Trade: {action.upper()} {quantity} shares of {symbol} at ${price}"
             
             memory_entry = MemoryEntry(
                 user_id=user_id,
                 memory_type=MemoryType.TRADING,
                 content=content,
                 context=f"Trading activity: {action} {symbol}",
                 importance=MemoryImportance.HIGH,
                 tags=["trade", symbol, action],
                 metadata={
                     "symbol": symbol,
                     "action": action,
                     "quantity": quantity,
                     "price": price,
                     **(context or {})
                 }
             )
             
             await self.intelligent_memory.add_memory(memory_entry)
             
             logger.info(f"Recorded trading activity for user {user_id}: {content}")
            
        except Exception as e:
            logger.error(f"Error recording trading activity: {e}")
    
    async def add_alert_activity(self, 
                               user_id: int, 
                               symbol: str, 
                               alert_type: str, 
                               condition: str,
                               context: Optional[Dict[str, Any]] = None):
        """Record alert activity in memory
        
        Args:
            user_id: User ID
            symbol: Stock symbol
            alert_type: Type of alert
            condition: Alert condition
            context: Additional context
        """
        try:
             content = f"Alert: {alert_type} for {symbol} when {condition}"
             
             memory_entry = MemoryEntry(
                 user_id=user_id,
                 memory_type=MemoryType.ALERT,
                 content=content,
                 context=f"Alert setup: {alert_type} for {symbol}",
                 importance=MemoryImportance.MEDIUM,
                 tags=["alert", symbol, alert_type],
                 metadata={
                     "symbol": symbol,
                     "alert_type": alert_type,
                     "condition": condition,
                     **(context or {})
                 }
             )
             
             await self.intelligent_memory.add_memory(memory_entry)
             
             logger.info(f"Recorded alert activity for user {user_id}: {content}")
            
        except Exception as e:
            logger.error(f"Error recording alert activity: {e}")
    
    async def get_memory_summary(self, user_id: int) -> Dict[str, Any]:
        """Get a comprehensive memory summary for a user
        
        Args:
            user_id: User ID
            
        Returns:
            Memory summary
        """
        try:
            # Get intelligent memory stats
            memory_stats = await self.intelligent_memory.get_memory_stats(user_id)
            
            # Get conversation stats
            conversation_stats = self.conversation_memory.get_session_stats(user_id)
            
            # Get recent insights
            recent_insights = await self.intelligent_memory.retrieve_memories(
                user_id=user_id,
                query="insight pattern behavior",
                memory_types=[MemoryType.INSIGHT],
                limit=5
            )
            
            # Get user preferences
            preferences = await self._get_user_preferences(user_id)
            
            summary = {
                "memory_stats": memory_stats,
                "conversation_stats": conversation_stats,
                "recent_insights": [
                    {
                        "content": insight.content,
                        "created_at": insight.created_at.isoformat()
                    }
                    for insight in recent_insights
                ],
                "preferences": preferences,
                "memory_health": {
                    "total_memories": memory_stats.get("total_memories", 0),
                    "active_session": user_id in self.conversation_memory.sessions,
                    "last_interaction": conversation_stats.get("last_activity", "Never")
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting memory summary: {e}")
            return {}
    
    # Private helper methods
    
    async def _extract_preferences(self, user_id: int, user_message: str, bot_response: str):
        """Extract preferences from conversation"""
        try:
            message_lower = user_message.lower()
            
            # Risk tolerance
            if any(word in message_lower for word in ["conservative", "safe", "low risk"]):
                await self._add_user_preference(user_id, "risk_tolerance", "conservative")
            elif any(word in message_lower for word in ["aggressive", "high risk", "risky"]):
                await self._add_user_preference(user_id, "risk_tolerance", "aggressive")
            
            # Investment horizon
            if any(phrase in message_lower for phrase in ["long term", "long-term", "years"]):
                await self._add_user_preference(user_id, "investment_horizon", "long_term")
            elif any(phrase in message_lower for phrase in ["short term", "short-term", "day trading"]):
                await self._add_user_preference(user_id, "investment_horizon", "short_term")
            
        except Exception as e:
            logger.error(f"Error extracting preferences: {e}")
    
    async def _add_user_preference(self, user_id: int, key: str, value: str):
        """Add user preference to memory"""
        try:
            await self.intelligent_memory.add_memory(
                user_id=user_id,
                content=f"User preference: {key} = {value}",
                memory_type=MemoryType.PREFERENCE,
                importance=MemoryImportance.HIGH,
                context={"preference_key": key, "preference_value": value},
                tags={"preference", key}
            )
        except Exception as e:
            logger.error(f"Error adding user preference: {e}")
    
    async def _get_user_preferences(self, user_id: int) -> Dict[str, str]:
        """Get user preferences from memory"""
        try:
            preference_memories = await self.intelligent_memory.retrieve_memories(
                user_id=user_id,
                query="preference",
                memory_types=[MemoryType.PREFERENCE],
                limit=50
            )
            
            preferences = {}
            for memory in preference_memories:
                if memory.context and "preference_key" in memory.context:
                    key = memory.context["preference_key"]
                    value = memory.context["preference_value"]
                    preferences[key] = value
            
            return preferences
            
        except Exception as e:
            logger.error(f"Error getting user preferences: {e}")
            return {}
    
    async def _get_user_patterns(self, user_id: int) -> List[str]:
        """Get user behavioral patterns"""
        try:
            pattern_memories = await self.intelligent_memory.retrieve_memories(
                user_id=user_id,
                query="pattern behavior insight",
                memory_types=[MemoryType.INSIGHT],
                limit=10
            )
            
            return [memory.content for memory in pattern_memories]
            
        except Exception as e:
            logger.error(f"Error getting user patterns: {e}")
            return []

# Global instance
enhanced_memory_service = EnhancedMemoryService()