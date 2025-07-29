"""Memory Integration Module
Integrates the enhanced memory system with existing bot components
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from functools import wraps

from enhanced_memory_service import enhanced_memory_service
from intelligent_memory_system import MemoryType, MemoryImportance
from logger import BotLogger
from performance_cache import time_operation

logger = BotLogger(__name__)

class MemoryIntegration:
    """Memory integration class that provides decorators and utilities
    for seamless memory integration across bot components
    """
    
    def __init__(self):
        """Initialize memory integration"""
        self.memory_service = enhanced_memory_service
        self.integration_enabled = True
        
        logger.info("Memory Integration initialized")
    
    def remember_interaction(self, 
                           memory_type: MemoryType = MemoryType.CONVERSATION,
                           importance: MemoryImportance = MemoryImportance.MEDIUM,
                           extract_context: bool = True):
        """Decorator to automatically remember user interactions
        
        Args:
            memory_type: Type of memory to store
            importance: Importance level
            extract_context: Whether to extract additional context
        """
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                try:
                    # Execute the original function
                    result = await func(*args, **kwargs)
                    
                    # Extract user_id and message from common patterns
                    user_id = None
                    user_message = ""
                    bot_response = ""
                    
                    # Try to extract from different argument patterns
                    if args:
                        # Pattern 1: update object with message
                        if hasattr(args[0], 'effective_user') and hasattr(args[0], 'message'):
                            user_id = args[0].effective_user.id
                            user_message = args[0].message.text or ""
                        # Pattern 2: context object
                        elif hasattr(args[0], 'user_data') and hasattr(args[0], 'bot'):
                            user_id = getattr(args[0], 'user_id', None)
                        # Pattern 3: direct user_id
                        elif isinstance(args[0], int):
                            user_id = args[0]
                            if len(args) > 1:
                                user_message = str(args[1])
                    
                    # Extract from kwargs
                    if not user_id:
                        user_id = kwargs.get('user_id')
                    if not user_message:
                        user_message = kwargs.get('message', kwargs.get('query', kwargs.get('text', "")))
                    
                    # Try to extract bot response
                    if isinstance(result, str):
                        bot_response = result
                    elif isinstance(result, dict) and 'response' in result:
                        bot_response = result['response']
                    elif hasattr(result, 'text'):
                        bot_response = result.text
                    
                    # Store the interaction if we have enough information
                    if user_id and self.integration_enabled:
                        context = {
                            'function_name': func.__name__,
                            'module': func.__module__,
                            'timestamp': datetime.utcnow().isoformat()
                        }
                        
                        if extract_context:
                            context.update(kwargs)
                        
                        await self.memory_service.add_interaction(
                            user_id=user_id,
                            user_message=user_message,
                            bot_response=bot_response,
                            context=context,
                            importance=importance
                        )
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"Error in memory integration decorator: {e}")
                    # Return original result even if memory fails
                    return await func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def remember_trading_activity(self, importance: MemoryImportance = MemoryImportance.HIGH):
        """Decorator to automatically remember trading activities
        
        Args:
            importance: Importance level
        """
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                try:
                    result = await func(*args, **kwargs)
                    
                    # Extract trading information
                    user_id = kwargs.get('user_id') or (args[0] if args and isinstance(args[0], int) else None)
                    symbol = kwargs.get('symbol', kwargs.get('ticker', ''))
                    action = kwargs.get('action', kwargs.get('trade_type', ''))
                    quantity = kwargs.get('quantity', kwargs.get('amount', 0))
                    price = kwargs.get('price', 0)
                    
                    if user_id and symbol and action and self.integration_enabled:
                        await self.memory_service.add_trading_activity(
                            user_id=user_id,
                            symbol=symbol,
                            action=action,
                            quantity=float(quantity) if quantity else 0,
                            price=float(price) if price else 0,
                            context={
                                'function': func.__name__,
                                'result': str(result)[:200]  # Truncate long results
                            }
                        )
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"Error in trading memory decorator: {e}")
                    return await func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def remember_alert_activity(self, importance: MemoryImportance = MemoryImportance.MEDIUM):
        """Decorator to automatically remember alert activities
        
        Args:
            importance: Importance level
        """
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                try:
                    result = await func(*args, **kwargs)
                    
                    # Extract alert information
                    user_id = kwargs.get('user_id') or (args[0] if args and isinstance(args[0], int) else None)
                    symbol = kwargs.get('symbol', kwargs.get('ticker', ''))
                    alert_type = kwargs.get('alert_type', kwargs.get('type', 'price_alert'))
                    condition = kwargs.get('condition', kwargs.get('trigger', ''))
                    
                    if user_id and symbol and self.integration_enabled:
                        await self.memory_service.add_alert_activity(
                            user_id=user_id,
                            symbol=symbol,
                            alert_type=alert_type,
                            condition=condition,
                            context={
                                'function': func.__name__,
                                'result': str(result)[:200]
                            }
                        )
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"Error in alert memory decorator: {e}")
                    return await func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    async def get_contextual_prompt_enhancement(self, 
                                              user_id: int, 
                                              current_query: str) -> Dict[str, Any]:
        """Get contextual information to enhance prompts and responses
        
        Args:
            user_id: User ID
            current_query: Current user query
            
        Returns:
            Dictionary with contextual enhancement data
        """
        try:
            context_data = await self.memory_service.get_contextual_response_data(
                user_id=user_id,
                current_query=current_query
            )
            
            # Format for prompt enhancement
            enhancement = {
                'user_context': context_data.get('intelligent_context', {}),
                'relevant_history': [],
                'user_preferences': context_data.get('user_preferences', {}),
                'behavioral_patterns': context_data.get('user_patterns', []),
                'recent_activities': []
            }
            
            # Process relevant memories
            relevant_memories = context_data.get('relevant_memories', [])
            for memory in relevant_memories[:5]:  # Limit to top 5
                if memory['memory_type'] in ['conversation', 'trade', 'alert']:
                    enhancement['relevant_history'].append({
                        'content': memory['content'],
                        'type': memory['memory_type'],
                        'importance': memory['importance']
                    })
                elif memory['memory_type'] in ['trade', 'alert']:
                    enhancement['recent_activities'].append({
                        'content': memory['content'],
                        'type': memory['memory_type']
                    })
            
            return enhancement
            
        except Exception as e:
            logger.error(f"Error getting contextual prompt enhancement: {e}")
            return {}
    
    def enable_integration(self):
        """Enable memory integration"""
        self.integration_enabled = True
        logger.info("Memory integration enabled")
    
    def disable_integration(self):
        """Disable memory integration"""
        self.integration_enabled = False
        logger.info("Memory integration disabled")
    
    def remember_preference(self, importance: MemoryImportance = MemoryImportance.HIGH):
        """Decorator to automatically remember user preferences
        
        Args:
            importance: Importance level
        """
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                try:
                    result = await func(*args, **kwargs)
                    
                    # Extract preference information
                    user_id = kwargs.get('user_id') or (args[0] if args and isinstance(args[0], int) else None)
                    preference_key = kwargs.get('preference_key', kwargs.get('key', ''))
                    preference_value = kwargs.get('preference_value', kwargs.get('value', ''))
                    
                    if user_id and preference_key and self.integration_enabled:
                        # Store preference in memory
                        from intelligent_memory_system import MemoryEntry
                        
                        memory_entry = MemoryEntry(
                            user_id=user_id,
                            memory_type=MemoryType.PREFERENCE,
                            content=f"User preference: {preference_key} = {preference_value}",
                            context=f"Preference setting: {preference_key}",
                            importance=importance,
                            tags=["preference", preference_key],
                            metadata={
                                "preference_key": preference_key,
                                "preference_value": preference_value,
                                "function": func.__name__
                            }
                        )
                        
                        await self.memory_service.intelligent_memory.add_memory(memory_entry)
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"Error in preference memory decorator: {e}")
                    return await func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def remember_error_context(self, importance: MemoryImportance = MemoryImportance.MEDIUM):
        """Decorator to automatically remember error contexts
        
        Args:
            importance: Importance level
        """
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                try:
                    result = await func(*args, **kwargs)
                    return result
                    
                except Exception as e:
                    # Extract error information
                    user_id = kwargs.get('user_id') or (args[0] if args and isinstance(args[0], int) else None)
                    error_type = type(e).__name__
                    error_message = str(e)
                    
                    if user_id and self.integration_enabled:
                        # Store error context in memory
                        from intelligent_memory_system import MemoryEntry
                        
                        memory_entry = MemoryEntry(
                            user_id=user_id,
                            memory_type=MemoryType.ERROR,
                            content=f"Error in {func.__name__}: {error_type} - {error_message}",
                            context=f"Function: {func.__name__}, Error: {error_type}",
                            importance=importance,
                            tags=["error", error_type.lower(), func.__name__],
                            metadata={
                                "function": func.__name__,
                                "error_type": error_type,
                                "error_message": error_message,
                                "args": str(args)[:100],
                                "kwargs": str(kwargs)[:100]
                            }
                        )
                        
                        try:
                            await self.memory_service.intelligent_memory.add_memory(memory_entry)
                        except Exception as mem_error:
                            logger.error(f"Failed to store error context: {mem_error}")
                    
                    # Re-raise the original exception
                    raise e
            
            return wrapper
        return decorator

# Global instance
memory_integration = MemoryIntegration()

# Convenience decorators
remember_interaction = memory_integration.remember_interaction
remember_trading_activity = memory_integration.remember_trading_activity
remember_alert_activity = memory_integration.remember_alert_activity
remember_preference = memory_integration.remember_preference
remember_error_context = memory_integration.remember_error_context