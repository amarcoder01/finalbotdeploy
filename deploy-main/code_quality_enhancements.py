#!/usr/bin/env python3
"""
Code Quality Enhancement Implementation
This module implements various improvements to enhance code quality and maintainability
"""

import asyncio
import functools
import time
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from dataclasses import dataclass
from enum import Enum
import logging
from contextlib import asynccontextmanager
import weakref

# Type definitions for better type safety
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

class ServiceStatus(Enum):
    """Enum for service status tracking"""
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"

@dataclass
class PerformanceMetrics:
    """Data class for tracking performance metrics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    last_request_time: Optional[float] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage"""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100

class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open"""
    pass

class CircuitBreaker:
    """Circuit breaker pattern implementation for resilient service calls"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half-open
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection"""
        if self.state == 'open':
            if time.time() - self.last_failure_time < self.recovery_timeout:
                raise CircuitBreakerError("Circuit breaker is open")
            else:
                self.state = 'half-open'
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        self.state = 'closed'
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = 'open'

def retry_with_backoff(max_retries: int = 3, backoff_factor: float = 1.0):
    """Decorator for retry logic with exponential backoff"""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        wait_time = backoff_factor * (2 ** attempt)
                        await asyncio.sleep(wait_time)
                    else:
                        raise last_exception
            return None
        return wrapper
    return decorator

def performance_monitor(func: F) -> F:
    """Decorator to monitor function performance"""
    metrics = PerformanceMetrics()
    
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        metrics.total_requests += 1
        
        try:
            result = await func(*args, **kwargs)
            metrics.successful_requests += 1
            return result
        except Exception as e:
            metrics.failed_requests += 1
            raise e
        finally:
            end_time = time.time()
            response_time = end_time - start_time
            metrics.last_request_time = end_time
            
            # Update average response time
            if metrics.total_requests == 1:
                metrics.average_response_time = response_time
            else:
                metrics.average_response_time = (
                    (metrics.average_response_time * (metrics.total_requests - 1) + response_time) 
                    / metrics.total_requests
                )
    
    wrapper.metrics = metrics
    return wrapper

class CacheManager:
    """Simple in-memory cache with TTL support"""
    
    def __init__(self, default_ttl: int = 300):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key in self.cache:
            entry = self.cache[key]
            if time.time() < entry['expires_at']:
                return entry['value']
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with TTL"""
        ttl = ttl or self.default_ttl
        self.cache[key] = {
            'value': value,
            'expires_at': time.time() + ttl
        }
    
    def clear(self) -> None:
        """Clear all cache entries"""
        self.cache.clear()
    
    def cleanup_expired(self) -> None:
        """Remove expired entries"""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.cache.items() 
            if current_time >= entry['expires_at']
        ]
        for key in expired_keys:
            del self.cache[key]

class ServiceHealthMonitor:
    """Monitor service health and status"""
    
    def __init__(self):
        self.services: Dict[str, Dict[str, Any]] = {}
    
    def register_service(self, name: str, check_func: Callable[[], bool]):
        """Register a service for health monitoring"""
        self.services[name] = {
            'check_func': check_func,
            'status': ServiceStatus.INITIALIZING,
            'last_check': None,
            'consecutive_failures': 0
        }
    
    async def check_service_health(self, name: str) -> ServiceStatus:
        """Check health of a specific service"""
        if name not in self.services:
            return ServiceStatus.UNHEALTHY
        
        service = self.services[name]
        try:
            is_healthy = await service['check_func']()
            if is_healthy:
                service['status'] = ServiceStatus.HEALTHY
                service['consecutive_failures'] = 0
            else:
                service['consecutive_failures'] += 1
                if service['consecutive_failures'] >= 3:
                    service['status'] = ServiceStatus.UNHEALTHY
                else:
                    service['status'] = ServiceStatus.DEGRADED
        except Exception:
            service['consecutive_failures'] += 1
            service['status'] = ServiceStatus.UNHEALTHY
        
        service['last_check'] = time.time()
        return service['status']
    
    async def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        health_report = {
            'overall_status': ServiceStatus.HEALTHY,
            'services': {},
            'timestamp': time.time()
        }
        
        unhealthy_count = 0
        for name in self.services:
            status = await self.check_service_health(name)
            health_report['services'][name] = status
            if status == ServiceStatus.UNHEALTHY:
                unhealthy_count += 1
        
        # Determine overall status
        total_services = len(self.services)
        if unhealthy_count == 0:
            health_report['overall_status'] = ServiceStatus.HEALTHY
        elif unhealthy_count < total_services / 2:
            health_report['overall_status'] = ServiceStatus.DEGRADED
        else:
            health_report['overall_status'] = ServiceStatus.UNHEALTHY
        
        return health_report

class ResourceManager:
    """Manage system resources and prevent memory leaks"""
    
    def __init__(self):
        self.active_connections = weakref.WeakSet()
        self.resource_limits = {
            'max_connections': 100,
            'max_cache_size': 1000,
            'max_memory_mb': 512
        }
    
    def register_connection(self, connection):
        """Register an active connection"""
        self.active_connections.add(connection)
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage statistics"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            'active_connections': len(self.active_connections),
            'memory_usage_mb': memory_info.rss / 1024 / 1024,
            'cpu_percent': process.cpu_percent(),
            'open_files': len(process.open_files()),
            'threads': process.num_threads()
        }
    
    def check_resource_limits(self) -> List[str]:
        """Check if any resource limits are exceeded"""
        warnings = []
        usage = self.get_resource_usage()
        
        if usage['active_connections'] > self.resource_limits['max_connections']:
            warnings.append(f"Too many connections: {usage['active_connections']}")
        
        if usage['memory_usage_mb'] > self.resource_limits['max_memory_mb']:
            warnings.append(f"High memory usage: {usage['memory_usage_mb']:.1f}MB")
        
        return warnings

@asynccontextmanager
async def timeout_context(seconds: float):
    """Async context manager for timeout handling"""
    try:
        async with asyncio.timeout(seconds):
            yield
    except asyncio.TimeoutError:
        raise asyncio.TimeoutError(f"Operation timed out after {seconds} seconds")

class ConfigValidator:
    """Validate configuration settings"""
    
    @staticmethod
    def validate_required_env_vars(required_vars: List[str]) -> List[str]:
        """Validate that required environment variables are set"""
        import os
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        return missing_vars
    
    @staticmethod
    def validate_api_keys(api_configs: Dict[str, str]) -> Dict[str, bool]:
        """Validate API key formats"""
        results = {}
        for service, key in api_configs.items():
            if not key:
                results[service] = False
            elif len(key) < 10:  # Basic length check
                results[service] = False
            else:
                results[service] = True
        return results

class ErrorHandler:
    """Centralized error handling and reporting"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.error_counts = {}
    
    def handle_error(self, error: Exception, context: str = "") -> str:
        """Handle and log errors with context"""
        error_type = type(error).__name__
        error_message = str(error)
        
        # Track error frequency
        if error_type not in self.error_counts:
            self.error_counts[error_type] = 0
        self.error_counts[error_type] += 1
        
        # Log with context
        self.logger.error(
            f"[{context}] {error_type}: {error_message} "
            f"(occurrence #{self.error_counts[error_type]})"
        )
        
        # Return user-friendly message
        return self._get_user_friendly_message(error_type, error_message)
    
    def _get_user_friendly_message(self, error_type: str, error_message: str) -> str:
        """Convert technical errors to user-friendly messages"""
        if "timeout" in error_message.lower():
            return "⏱️ Request timed out. Please try again."
        elif "rate limit" in error_message.lower():
            return "🚦 Rate limit exceeded. Please wait a moment and try again."
        elif "network" in error_message.lower():
            return "🌐 Network error. Please check your connection and try again."
        elif "api key" in error_message.lower():
            return "🔑 API configuration issue. Please contact support."
        else:
            return "❌ An unexpected error occurred. Please try again later."
    
    def get_error_summary(self) -> Dict[str, int]:
        """Get summary of error occurrences"""
        return self.error_counts.copy()

# Global instances for easy access
cache_manager = CacheManager()
health_monitor = ServiceHealthMonitor()
resource_manager = ResourceManager()