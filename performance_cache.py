#!/usr/bin/env python3
"""
Performance Cache System for High Traffic Optimization
Implements intelligent caching, response compression, and cold start reduction
"""

import asyncio
import time
import json
import gzip
import hashlib
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, asdict
from functools import wraps, lru_cache
from collections import defaultdict
import threading
from logger import logger

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    data: Any
    timestamp: float
    access_count: int = 0
    last_access: float = 0
    compressed: bool = False
    size_bytes: int = 0

class PerformanceCache:
    """High-performance caching system with intelligent eviction"""
    
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, CacheEntry] = {}
        self.access_times: Dict[str, float] = {}
        self.lock = threading.RLock()
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
        
        # Start cleanup worker
        self._start_cleanup_worker()
        
        logger.info(f"Performance cache initialized: max_size={max_size}, ttl={ttl_seconds}s")
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items()) if kwargs else {}
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _compress_data(self, data: Any) -> Tuple[bytes, bool]:
        """Compress data if beneficial"""
        try:
            json_str = json.dumps(data, default=str)
            json_bytes = json_str.encode('utf-8')
            
            # Only compress if data is large enough
            if len(json_bytes) > 1024:  # 1KB threshold
                compressed = gzip.compress(json_bytes)
                if len(compressed) < len(json_bytes) * 0.8:  # 20% compression benefit
                    return compressed, True
            
            return json_bytes, False
        except Exception:
            return str(data).encode('utf-8'), False
    
    def _decompress_data(self, data: bytes, compressed: bool) -> Any:
        """Decompress cached data"""
        try:
            if compressed:
                decompressed = gzip.decompress(data)
                return json.loads(decompressed.decode('utf-8'))
            else:
                return json.loads(data.decode('utf-8'))
        except Exception:
            return data.decode('utf-8')
    
    def _evict_lru(self):
        """Evict least recently used items"""
        if len(self.cache) <= self.max_size:
            return
        
        # Sort by last access time
        sorted_items = sorted(
            self.cache.items(),
            key=lambda x: x[1].last_access
        )
        
        # Remove oldest 10% of items
        evict_count = max(1, len(self.cache) // 10)
        for i in range(evict_count):
            key, _ = sorted_items[i]
            del self.cache[key]
            self.access_times.pop(key, None)
            self.eviction_count += 1
        
        logger.debug(f"Evicted {evict_count} cache entries")
    
    def _cleanup_expired(self):
        """Remove expired cache entries"""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self.cache.items():
            if current_time - entry.timestamp > self.ttl_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
            self.access_times.pop(key, None)
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def _start_cleanup_worker(self):
        """Start background cleanup worker"""
        def cleanup_worker():
            while True:
                time.sleep(300)  # Cleanup every 5 minutes
                with self.lock:
                    self._cleanup_expired()
                    if len(self.cache) > self.max_size:
                        self._evict_lru()
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        with self.lock:
            if key not in self.cache:
                self.miss_count += 1
                return None
            
            entry = self.cache[key]
            current_time = time.time()
            
            # Check if expired
            if current_time - entry.timestamp > self.ttl_seconds:
                del self.cache[key]
                self.access_times.pop(key, None)
                self.miss_count += 1
                return None
            
            # Update access statistics
            entry.access_count += 1
            entry.last_access = current_time
            self.access_times[key] = current_time
            self.hit_count += 1
            
            # Decompress if needed
            if isinstance(entry.data, bytes):
                return self._decompress_data(entry.data, entry.compressed)
            
            return entry.data
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set item in cache"""
        with self.lock:
            current_time = time.time()
            
            # Compress data if beneficial
            if isinstance(value, (dict, list, str)) and len(str(value)) > 100:
                compressed_data, is_compressed = self._compress_data(value)
                data = compressed_data
                size_bytes = len(compressed_data)
            else:
                data = value
                is_compressed = False
                size_bytes = len(str(value).encode('utf-8'))
            
            entry = CacheEntry(
                data=data,
                timestamp=current_time,
                access_count=1,
                last_access=current_time,
                compressed=is_compressed,
                size_bytes=size_bytes
            )
            
            self.cache[key] = entry
            self.access_times[key] = current_time
            
            # Evict if necessary
            if len(self.cache) > self.max_size:
                self._evict_lru()
    
    def delete(self, key: str) -> bool:
        """Delete item from cache"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                self.access_times.pop(key, None)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_requests = self.hit_count + self.miss_count
            hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0
            
            total_size = sum(entry.size_bytes for entry in self.cache.values())
            
            return {
                'entries': len(self.cache),
                'max_size': self.max_size,
                'hit_count': self.hit_count,
                'miss_count': self.miss_count,
                'hits': self.hit_count,  # Alias for compatibility
                'misses': self.miss_count,  # Alias for compatibility
                'hit_rate_percent': round(hit_rate, 2),
                'eviction_count': self.eviction_count,
                'total_size_bytes': total_size,
                'average_size_bytes': round(total_size / len(self.cache)) if self.cache else 0
            }

class ResponseCache:
    """Specialized cache for common bot responses"""
    
    def __init__(self):
        self.cache = PerformanceCache(max_size=5000, ttl_seconds=1800)  # 30 minutes
        self.response_templates = {
            'price_not_found': "❌ Sorry, I couldn't find price data for {symbol}. Please check the symbol and try again.",
            'analysis_error': "⚠️ Unable to perform analysis for {symbol} at the moment. Please try again later.",
            'market_closed': "🕐 Markets are currently closed. Showing last available data for {symbol}.",
            'rate_limit': "⏳ Please wait a moment before making another request.",
            'invalid_symbol': "❌ '{symbol}' is not a valid stock symbol. Please check and try again.",
            'help_menu': self._get_help_menu(),
            'welcome_message': "👋 Welcome to AI Trading Companion! Use /help to see available commands."
        }
        
        # Pre-cache common responses
        self._precache_responses()
        
        logger.info("Response cache initialized with templates")
    
    def _get_help_menu(self) -> str:
        """Generate help menu"""
        return """
🤖 **AI Trading Companion Commands**

📊 **Market Data:**
• `/price SYMBOL` - Get current price
• `/chart SYMBOL` - Generate price chart
• `/analyze SYMBOL` - AI market analysis
• `/deep_analysis SYMBOL` - Deep learning analysis

📈 **Trading:**
• `/trade` - Execute trades
• `/portfolio` - View portfolio
• `/alerts` - Manage alerts

🔧 **Utilities:**
• `/help` - Show this menu
• `/settings` - User settings

Type any command to get started! 🚀
"""
    
    def _precache_responses(self):
        """Pre-cache common response templates"""
        for template_name, template in self.response_templates.items():
            self.cache.set(f"template_{template_name}", template, ttl=7200)  # 2 hours
    
    def get_template(self, template_name: str, **kwargs) -> str:
        """Get response template with formatting"""
        template = self.cache.get(f"template_{template_name}")
        if template is None:
            template = self.response_templates.get(template_name, "Template not found")
            self.cache.set(f"template_{template_name}", template)
        
        try:
            return template.format(**kwargs)
        except KeyError:
            return template
    
    def cache_response(self, key: str, response: str, ttl: int = 1800) -> None:
        """Cache a response"""
        self.cache.set(key, response, ttl)
    
    def get_response(self, key: str) -> Optional[str]:
        """Get cached response"""
        return self.cache.get(key)
    
    def get(self, key: str) -> Optional[str]:
        """Get cached response (alias for get_response)"""
        return self.get_response(key)
    
    def set(self, key: str, value: str, ttl: int = 1800) -> None:
        """Set cached response (alias for cache_response)"""
        self.cache_response(key, value, ttl)

class ConnectionPool:
    """Connection pool for external API calls"""
    
    def __init__(self, max_connections: int = 50):
        self.max_connections = max_connections
        self.active_connections = 0
        self.semaphore = asyncio.Semaphore(max_connections)
        self.connection_stats = defaultdict(int)
        
        logger.info(f"Connection pool initialized: max_connections={max_connections}")
    
    async def acquire(self, service_name: str = "default"):
        """Acquire connection from pool"""
        await self.semaphore.acquire()
        self.active_connections += 1
        self.connection_stats[service_name] += 1
        return ConnectionContext(self, service_name)
    
    def release(self, service_name: str = "default"):
        """Release connection back to pool"""
        self.active_connections -= 1
        self.semaphore.release()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        return {
            'max_connections': self.max_connections,
            'total_connections': self.max_connections,  # Alias for compatibility
            'active_connections': self.active_connections,
            'available_connections': self.max_connections - self.active_connections,
            'service_usage': dict(self.connection_stats)
        }
    
    def get_connection(self, service_name: str = "default"):
        """Get connection (synchronous version for compatibility)"""
        return SyncConnectionContext(self, service_name)

class ConnectionContext:
    """Context manager for connection pool"""
    
    def __init__(self, pool: ConnectionPool, service_name: str):
        self.pool = pool
        self.service_name = service_name
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.pool.release(self.service_name)

class SyncConnectionContext:
    """Synchronous context manager for connection pool"""
    
    def __init__(self, pool: ConnectionPool, service_name: str):
        self.pool = pool
        self.service_name = service_name
        self.acquired = False
    
    def __enter__(self):
        # Simulate acquiring connection (synchronous)
        if self.pool.active_connections < self.pool.max_connections:
            self.pool.active_connections += 1
            self.pool.connection_stats[self.service_name] += 1
            self.acquired = True
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.acquired:
            self.pool.active_connections -= 1
            self.acquired = False

# Global instances
performance_cache = PerformanceCache()
response_cache = ResponseCache()
connection_pool = ConnectionPool()

# Decorators for caching
def cache_result(ttl: int = 3600, key_prefix: str = ""):
    """Decorator to cache function results"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{key_prefix}{func.__name__}_{performance_cache._generate_key(*args, **kwargs)}"
            
            # Try to get from cache
            cached_result = performance_cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            performance_cache.set(cache_key, result, ttl)
            logger.debug(f"Cached result for {func.__name__}")
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{key_prefix}{func.__name__}_{performance_cache._generate_key(*args, **kwargs)}"
            
            # Try to get from cache
            cached_result = performance_cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            performance_cache.set(cache_key, result, ttl)
            logger.debug(f"Cached result for {func.__name__}")
            return result
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

def with_connection_pool(service_name: str = "default"):
    """Decorator to use connection pool"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async with await connection_pool.acquire(service_name):
                return await func(*args, **kwargs)
        return wrapper
    return decorator

def time_operation(operation_name: str = ""):
    """Decorator to time function execution"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.debug(f"{operation_name or func.__name__} completed in {execution_time:.3f}s")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"{operation_name or func.__name__} failed after {execution_time:.3f}s: {e}")
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.debug(f"{operation_name or func.__name__} completed in {execution_time:.3f}s")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"{operation_name or func.__name__} failed after {execution_time:.3f}s: {e}")
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

# Utility functions
def get_cache_stats() -> Dict[str, Any]:
    """Get comprehensive cache statistics"""
    return {
        'performance_cache': performance_cache.get_stats(),
        'response_cache': response_cache.cache.get_stats(),
        'connection_pool': connection_pool.get_stats()
    }

def clear_all_caches():
    """Clear all caches"""
    performance_cache.clear()
    response_cache.cache.clear()
    logger.info("All caches cleared")

# Pre-loading utilities
class PreLoader:
    """Pre-load common data to reduce cold start time"""
    
    def __init__(self):
        self.preloaded_data = {}
        self.preload_tasks = []
    
    async def preload_market_data(self):
        """Pre-load common market data"""
        try:
            # Pre-load popular symbols
            popular_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX']
            
            for symbol in popular_symbols:
                # This would be replaced with actual market data loading
                cache_key = f"preload_price_{symbol}"
                performance_cache.set(cache_key, f"Preloaded data for {symbol}", ttl=300)
            
            logger.info(f"Pre-loaded market data for {len(popular_symbols)} symbols")
        except Exception as e:
            logger.error(f"Failed to preload market data: {e}")
    
    async def preload_ai_models(self):
        """Pre-load AI model components"""
        try:
            # Pre-load model metadata
            model_info = {
                'openai_models': ['gpt-4o-mini', 'gpt-4o'],
                'deep_learning_models': ['lstm', 'transformer'],
                'last_updated': time.time()
            }
            
            performance_cache.set("preload_ai_models", model_info, ttl=3600)
            logger.info("Pre-loaded AI model information")
        except Exception as e:
            logger.error(f"Failed to preload AI models: {e}")
    
    async def start_preloading(self):
        """Start all preloading tasks"""
        self.preload_tasks = [
            asyncio.create_task(self.preload_market_data()),
            asyncio.create_task(self.preload_ai_models())
        ]
        
        await asyncio.gather(*self.preload_tasks, return_exceptions=True)
        logger.info("Preloading completed")

# Global preloader instance
preloader = PreLoader()

logger.info("Performance cache system initialized")