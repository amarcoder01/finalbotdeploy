#!/usr/bin/env python3
"""
Enhanced Service Layer
Provides improved service implementations with better error handling, caching, and async support
"""

import asyncio
import aiohttp
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
import logging
from pathlib import Path
import hashlib
from collections import defaultdict
import threading
from contextlib import asynccontextmanager

from enhanced_monitoring import (
    performance_monitor, record_error, AlertLevel, 
    CircuitBreaker, monitoring_system
)
from enhanced_config import config

class CacheEntry:
    """Cache entry with TTL support"""
    
    def __init__(self, data: Any, ttl_seconds: int = 300):
        self.data = data
        self.created_at = time.time()
        self.ttl_seconds = ttl_seconds
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        return time.time() - self.created_at > self.ttl_seconds
    
    def get_data(self) -> Optional[Any]:
        """Get data if not expired"""
        return None if self.is_expired() else self.data

class AsyncCache:
    """Thread-safe async cache with TTL support"""
    
    def __init__(self, default_ttl: int = 300, max_size: int = 1000):
        self.default_ttl = default_ttl
        self.max_size = max_size
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
        self._access_times: Dict[str, float] = {}
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        async with self._lock:
            entry = self._cache.get(key)
            if entry and not entry.is_expired():
                self._access_times[key] = time.time()
                return entry.get_data()
            elif entry:
                # Remove expired entry
                del self._cache[key]
                self._access_times.pop(key, None)
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache"""
        async with self._lock:
            # Clean up if at max size
            if len(self._cache) >= self.max_size:
                await self._evict_lru()
            
            ttl = ttl or self.default_ttl
            self._cache[key] = CacheEntry(value, ttl)
            self._access_times[key] = time.time()
    
    async def delete(self, key: str) -> None:
        """Delete key from cache"""
        async with self._lock:
            self._cache.pop(key, None)
            self._access_times.pop(key, None)
    
    async def clear(self) -> None:
        """Clear all cache entries"""
        async with self._lock:
            self._cache.clear()
            self._access_times.clear()
    
    async def _evict_lru(self) -> None:
        """Evict least recently used entry"""
        if not self._access_times:
            return
        
        lru_key = min(self._access_times.items(), key=lambda x: x[1])[0]
        del self._cache[lru_key]
        del self._access_times[lru_key]
    
    async def cleanup_expired(self) -> int:
        """Remove expired entries and return count"""
        async with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items() 
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                del self._cache[key]
                self._access_times.pop(key, None)
            
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_entries = len(self._cache)
        expired_count = sum(1 for entry in self._cache.values() if entry.is_expired())
        
        return {
            "total_entries": total_entries,
            "active_entries": total_entries - expired_count,
            "expired_entries": expired_count,
            "max_size": self.max_size,
            "utilization_percent": (total_entries / self.max_size) * 100
        }

class BaseEnhancedService(ABC):
    """Base class for enhanced services"""
    
    def __init__(self, service_name: str, cache_ttl: int = 300):
        self.service_name = service_name
        self.logger = logging.getLogger(f"{__name__}.{service_name}")
        self.cache = AsyncCache(default_ttl=cache_ttl)
        self.circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        self.request_count = 0
        self.error_count = 0
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Register health check
        monitoring_system.health_monitor.register_service(
            service_name, self._health_check, check_interval=60
        )
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self._initialize_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self._cleanup_session()
    
    async def _initialize_session(self):
        """Initialize HTTP session"""
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=config.performance.request_timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
    
    async def _cleanup_session(self):
        """Cleanup HTTP session"""
        if self._session:
            await self._session.close()
            self._session = None
    
    def _health_check(self) -> bool:
        """Basic health check implementation"""
        try:
            # Check error rate
            if self.request_count > 0:
                error_rate = (self.error_count / self.request_count) * 100
                if error_rate > 50:  # More than 50% error rate
                    return False
            
            # Check circuit breaker state
            if hasattr(self.circuit_breaker, 'state') and self.circuit_breaker.state == 'OPEN':
                return False
            
            return True
        except Exception:
            return False
    
    def _generate_cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_data = f"{self.service_name}:{args}:{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def _make_request(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request with error handling"""
        if not self._session:
            await self._initialize_session()
        
        try:
            async with self._session.request(method, url, **kwargs) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            self.error_count += 1
            record_error(e, context={"url": url, "method": method}, service_name=self.service_name)
            raise
        finally:
            self.request_count += 1
    
    @abstractmethod
    async def get_data(self, *args, **kwargs) -> Any:
        """Abstract method for getting data"""
        pass
    
    async def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        cache_stats = self.cache.get_stats()
        error_rate = (self.error_count / self.request_count * 100) if self.request_count > 0 else 0
        
        return {
            "service_name": self.service_name,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate_percent": error_rate,
            "circuit_breaker_state": getattr(self.circuit_breaker, 'state', 'UNKNOWN'),
            "cache_stats": cache_stats
        }

class EnhancedMarketDataService(BaseEnhancedService):
    """Enhanced market data service with multiple providers"""
    
    def __init__(self):
        super().__init__("market_data", cache_ttl=60)  # 1 minute cache for market data
        self.providers = {
            "alpaca": self._get_alpaca_data,
            "alpha_vantage": self._get_alpha_vantage_data,
            "yahoo": self._get_yahoo_data
        }
        self.provider_priority = ["alpaca", "alpha_vantage", "yahoo"]
    
    @performance_monitor("market_data")
    async def get_data(self, symbol: str, timeframe: str = "1D", limit: int = 100) -> Dict[str, Any]:
        """Get market data with fallback providers"""
        cache_key = self._generate_cache_key(symbol, timeframe, limit)
        
        # Try cache first
        cached_data = await self.cache.get(cache_key)
        if cached_data:
            return cached_data
        
        # Try providers in order
        last_error = None
        for provider_name in self.provider_priority:
            try:
                provider_func = self.providers[provider_name]
                data = await self._with_circuit_breaker(provider_func, symbol, timeframe, limit)
                
                # Cache successful result
                await self.cache.set(cache_key, data, ttl=60)
                return data
                
            except Exception as e:
                last_error = e
                self.logger.warning(f"Provider {provider_name} failed for {symbol}: {e}")
                continue
        
        # All providers failed
        error_msg = f"All market data providers failed for {symbol}"
        record_error(Exception(error_msg), 
                    context={"symbol": symbol, "last_error": str(last_error)},
                    service_name=self.service_name, severity=AlertLevel.ERROR)
        raise Exception(error_msg)
    
    async def _with_circuit_breaker(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.circuit_breaker(func), *args, **kwargs
        )
    
    async def _get_alpaca_data(self, symbol: str, timeframe: str, limit: int) -> Dict[str, Any]:
        """Get data from Alpaca API"""
        api_config = config.get_api_config("alpaca")
        if not api_config or not api_config.is_valid():
            raise Exception("Alpaca API not configured")
        
        headers = {
            "APCA-API-KEY-ID": api_config.api_key,
            "APCA-API-SECRET-KEY": config.ALPACA_API_SECRET
        }
        
        url = f"{api_config.base_url}/v2/stocks/{symbol}/bars"
        params = {
            "timeframe": timeframe,
            "limit": limit,
            "asof": datetime.utcnow().isoformat()
        }
        
        response = await self._make_request("GET", url, headers=headers, params=params)
        return self._normalize_alpaca_data(response)
    
    async def _get_alpha_vantage_data(self, symbol: str, timeframe: str, limit: int) -> Dict[str, Any]:
        """Get data from Alpha Vantage API"""
        api_config = config.get_api_config("alpha_vantage")
        if not api_config or not api_config.is_valid():
            raise Exception("Alpha Vantage API not configured")
        
        function = "TIME_SERIES_DAILY" if timeframe == "1D" else "TIME_SERIES_INTRADAY"
        url = f"{api_config.base_url}/query"
        params = {
            "function": function,
            "symbol": symbol,
            "apikey": api_config.api_key,
            "outputsize": "compact"
        }
        
        if function == "TIME_SERIES_INTRADAY":
            params["interval"] = "5min"
        
        response = await self._make_request("GET", url, params=params)
        return self._normalize_alpha_vantage_data(response, limit)
    
    async def _get_yahoo_data(self, symbol: str, timeframe: str, limit: int) -> Dict[str, Any]:
        """Get data from Yahoo Finance (simplified implementation)"""
        # This would require a Yahoo Finance library or API
        # For now, return mock data
        raise Exception("Yahoo Finance provider not implemented")
    
    def _normalize_alpaca_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Alpaca API response"""
        if "bars" not in data:
            raise Exception("Invalid Alpaca response format")
        
        bars = data["bars"]
        return {
            "symbol": data.get("symbol", ""),
            "data": [
                {
                    "timestamp": bar["t"],
                    "open": bar["o"],
                    "high": bar["h"],
                    "low": bar["l"],
                    "close": bar["c"],
                    "volume": bar["v"]
                }
                for bar in bars
            ],
            "provider": "alpaca",
            "retrieved_at": datetime.utcnow().isoformat()
        }
    
    def _normalize_alpha_vantage_data(self, data: Dict[str, Any], limit: int) -> Dict[str, Any]:
        """Normalize Alpha Vantage API response"""
        # Find the time series key
        time_series_key = None
        for key in data.keys():
            if "Time Series" in key:
                time_series_key = key
                break
        
        if not time_series_key:
            raise Exception("Invalid Alpha Vantage response format")
        
        time_series = data[time_series_key]
        symbol = data.get("Meta Data", {}).get("2. Symbol", "")
        
        # Convert to normalized format
        normalized_data = []
        for timestamp, values in list(time_series.items())[:limit]:
            normalized_data.append({
                "timestamp": timestamp,
                "open": float(values["1. open"]),
                "high": float(values["2. high"]),
                "low": float(values["3. low"]),
                "close": float(values["4. close"]),
                "volume": int(values["5. volume"])
            })
        
        return {
            "symbol": symbol,
            "data": normalized_data,
            "provider": "alpha_vantage",
            "retrieved_at": datetime.utcnow().isoformat()
        }

class EnhancedOpenAIService(BaseEnhancedService):
    """Enhanced OpenAI service with conversation memory and rate limiting"""
    
    def __init__(self):
        super().__init__("openai", cache_ttl=0)  # No caching for AI responses
        self.conversation_memory: Dict[int, List[Dict[str, str]]] = defaultdict(list)
        self.rate_limits: Dict[int, List[float]] = defaultdict(list)
        self.max_conversation_length = 20
        self.rate_limit_window = 60  # 1 minute
        self.max_requests_per_window = 10
    
    @performance_monitor("openai")
    async def get_data(self, user_id: int, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get AI response with rate limiting and conversation memory"""
        # Check rate limit
        if not self._check_rate_limit(user_id):
            raise Exception(f"Rate limit exceeded for user {user_id}")
        
        # Update conversation memory
        self._update_conversation(user_id, "user", message)
        
        try:
            # Prepare messages for API
            messages = self._prepare_messages(user_id, context)
            
            # Make API request
            response = await self._make_openai_request(messages)
            
            # Update conversation memory with response
            assistant_message = response["choices"][0]["message"]["content"]
            self._update_conversation(user_id, "assistant", assistant_message)
            
            return {
                "response": assistant_message,
                "usage": response.get("usage", {}),
                "model": response.get("model", config.OPENAI_MODEL),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            record_error(e, context={"user_id": user_id, "message_length": len(message)},
                        service_name=self.service_name)
            raise
    
    def _check_rate_limit(self, user_id: int) -> bool:
        """Check if user is within rate limits"""
        now = time.time()
        user_requests = self.rate_limits[user_id]
        
        # Remove old requests outside the window
        user_requests[:] = [req_time for req_time in user_requests 
                           if now - req_time < self.rate_limit_window]
        
        # Check if under limit
        if len(user_requests) >= self.max_requests_per_window:
            return False
        
        # Add current request
        user_requests.append(now)
        return True
    
    def _update_conversation(self, user_id: int, role: str, content: str):
        """Update conversation memory"""
        conversation = self.conversation_memory[user_id]
        conversation.append({"role": role, "content": content})
        
        # Trim conversation if too long
        if len(conversation) > self.max_conversation_length:
            # Keep system message (if any) and recent messages
            system_messages = [msg for msg in conversation if msg["role"] == "system"]
            recent_messages = conversation[-(self.max_conversation_length - len(system_messages)):]
            self.conversation_memory[user_id] = system_messages + recent_messages
    
    def _prepare_messages(self, user_id: int, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
        """Prepare messages for OpenAI API"""
        messages = []
        
        # Add system message with context
        system_content = "You are a helpful AI trading assistant."
        if context:
            system_content += f" Context: {json.dumps(context)}"
        
        messages.append({"role": "system", "content": system_content})
        
        # Add conversation history
        messages.extend(self.conversation_memory[user_id])
        
        return messages
    
    async def _make_openai_request(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Make request to OpenAI API"""
        api_config = config.get_api_config("openai")
        if not api_config or not api_config.is_valid():
            raise Exception("OpenAI API not configured")
        
        headers = {
            "Authorization": f"Bearer {api_config.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": config.OPENAI_MODEL,
            "messages": messages,
            "max_tokens": config.OPENAI_MAX_TOKENS,
            "temperature": config.OPENAI_TEMPERATURE
        }
        
        url = f"{api_config.base_url}/v1/chat/completions"
        return await self._make_request("POST", url, headers=headers, json=payload)
    
    def clear_conversation(self, user_id: int):
        """Clear conversation memory for user"""
        self.conversation_memory.pop(user_id, None)
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get conversation statistics"""
        total_conversations = len(self.conversation_memory)
        total_messages = sum(len(conv) for conv in self.conversation_memory.values())
        avg_conversation_length = total_messages / total_conversations if total_conversations > 0 else 0
        
        return {
            "total_conversations": total_conversations,
            "total_messages": total_messages,
            "avg_conversation_length": avg_conversation_length,
            "active_rate_limited_users": len(self.rate_limits)
        }

class ServiceManager:
    """Manage all enhanced services"""
    
    def __init__(self):
        self.services: Dict[str, BaseEnhancedService] = {}
        self.logger = logging.getLogger(__name__)
    
    async def register_service(self, service: BaseEnhancedService):
        """Register a service"""
        self.services[service.service_name] = service
        await service._initialize_session()
        self.logger.info(f"Registered service: {service.service_name}")
    
    async def get_service(self, service_name: str) -> Optional[BaseEnhancedService]:
        """Get a registered service"""
        return self.services.get(service_name)
    
    async def cleanup_all_services(self):
        """Cleanup all services"""
        for service in self.services.values():
            try:
                await service._cleanup_session()
                # Cleanup expired cache entries
                expired_count = await service.cache.cleanup_expired()
                if expired_count > 0:
                    self.logger.info(f"Cleaned up {expired_count} expired cache entries for {service.service_name}")
            except Exception as e:
                self.logger.error(f"Error cleaning up service {service.service_name}: {e}")
    
    async def get_all_service_stats(self) -> Dict[str, Any]:
        """Get statistics for all services"""
        stats = {}
        for service_name, service in self.services.items():
            try:
                stats[service_name] = await service.get_service_stats()
            except Exception as e:
                stats[service_name] = {"error": str(e)}
        return stats
    
    async def health_check_all_services(self) -> Dict[str, bool]:
        """Perform health check on all services"""
        health_status = {}
        for service_name, service in self.services.items():
            try:
                health_status[service_name] = service._health_check()
            except Exception as e:
                health_status[service_name] = False
                self.logger.error(f"Health check failed for {service_name}: {e}")
        return health_status

# Global service manager
service_manager = ServiceManager()

# Convenience functions for backward compatibility
async def get_market_data(symbol: str, timeframe: str = "1D", limit: int = 100) -> Dict[str, Any]:
    """Get market data using enhanced service"""
    market_service = await service_manager.get_service("market_data")
    if not market_service:
        market_service = EnhancedMarketDataService()
        await service_manager.register_service(market_service)
    
    return await market_service.get_data(symbol, timeframe, limit)

async def get_ai_response(user_id: int, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get AI response using enhanced service"""
    openai_service = await service_manager.get_service("openai")
    if not openai_service:
        openai_service = EnhancedOpenAIService()
        await service_manager.register_service(openai_service)
    
    return await openai_service.get_data(user_id, message, context)