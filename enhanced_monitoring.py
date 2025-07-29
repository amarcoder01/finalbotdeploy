#!/usr/bin/env python3
"""
Enhanced Monitoring and Error Handling System
Provides comprehensive monitoring, error tracking, and performance metrics
"""

import asyncio
import time
import traceback
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import logging
import json
from pathlib import Path
import functools
from contextlib import asynccontextmanager, contextmanager

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class ServiceStatus(Enum):
    """Service health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    memory_percent: float = 0.0
    active_threads: int = 0
    response_time_ms: float = 0.0
    requests_per_minute: int = 0
    error_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "cpu_percent": self.cpu_percent,
            "memory_mb": self.memory_mb,
            "memory_percent": self.memory_percent,
            "active_threads": self.active_threads,
            "response_time_ms": self.response_time_ms,
            "requests_per_minute": self.requests_per_minute,
            "error_rate": self.error_rate
        }

@dataclass
class ErrorInfo:
    """Error information structure"""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    error_type: str = ""
    error_message: str = ""
    traceback_str: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[int] = None
    service_name: str = ""
    severity: AlertLevel = AlertLevel.ERROR
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "error_type": self.error_type,
            "error_message": self.error_message,
            "traceback": self.traceback_str,
            "context": self.context,
            "user_id": self.user_id,
            "service_name": self.service_name,
            "severity": self.severity.value
        }

class CircuitBreaker:
    """Circuit breaker pattern implementation"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60, expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()
    
    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self._lock:
                if self.state == 'OPEN':
                    if self._should_attempt_reset():
                        self.state = 'HALF_OPEN'
                    else:
                        raise Exception(f"Circuit breaker is OPEN. Service unavailable.")
                
                try:
                    result = func(*args, **kwargs)
                    self._on_success()
                    return result
                except self.expected_exception as e:
                    self._on_failure()
                    raise e
        
        return wrapper
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        return (time.time() - self.last_failure_time) >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful execution"""
        self.failure_count = 0
        self.state = 'CLOSED'
    
    def _on_failure(self):
        """Handle failed execution"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'

class PerformanceMonitor:
    """Performance monitoring and metrics collection"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history: deque = deque(maxlen=max_history)
        self.request_times: deque = deque(maxlen=100)
        self.error_counts = defaultdict(int)
        self.request_counts = defaultdict(int)
        self._lock = threading.Lock()
        self.start_time = datetime.utcnow()
    
    def record_request(self, duration_ms: float, success: bool = True, service_name: str = "unknown"):
        """Record a request with its duration and success status"""
        with self._lock:
            self.request_times.append(duration_ms)
            minute_key = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
            self.request_counts[minute_key] += 1
            
            if not success:
                self.error_counts[service_name] += 1
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current system performance metrics"""
        process = psutil.Process()
        
        # Calculate average response time
        avg_response_time = 0.0
        if self.request_times:
            avg_response_time = sum(self.request_times) / len(self.request_times)
        
        # Calculate requests per minute
        current_minute = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
        requests_per_minute = self.request_counts.get(current_minute, 0)
        
        # Calculate error rate
        total_errors = sum(self.error_counts.values())
        total_requests = sum(self.request_counts.values())
        error_rate = (total_errors / total_requests * 100) if total_requests > 0 else 0.0
        
        metrics = PerformanceMetrics(
            cpu_percent=process.cpu_percent(),
            memory_mb=process.memory_info().rss / 1024 / 1024,
            memory_percent=process.memory_percent(),
            active_threads=threading.active_count(),
            response_time_ms=avg_response_time,
            requests_per_minute=requests_per_minute,
            error_rate=error_rate
        )
        
        with self._lock:
            self.metrics_history.append(metrics)
        
        return metrics
    
    def get_metrics_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get metrics summary for the specified time period"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {"message": "No metrics available for the specified period"}
        
        return {
            "period_hours": hours,
            "sample_count": len(recent_metrics),
            "avg_cpu_percent": sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics),
            "avg_memory_mb": sum(m.memory_mb for m in recent_metrics) / len(recent_metrics),
            "max_memory_mb": max(m.memory_mb for m in recent_metrics),
            "avg_response_time_ms": sum(m.response_time_ms for m in recent_metrics) / len(recent_metrics),
            "max_response_time_ms": max(m.response_time_ms for m in recent_metrics),
            "avg_error_rate": sum(m.error_rate for m in recent_metrics) / len(recent_metrics),
            "uptime_hours": (datetime.utcnow() - self.start_time).total_seconds() / 3600
        }

class ErrorTracker:
    """Error tracking and analysis system"""
    
    def __init__(self, max_errors: int = 1000):
        self.max_errors = max_errors
        self.errors: deque = deque(maxlen=max_errors)
        self.error_patterns = defaultdict(int)
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def record_error(self, error: Exception, context: Dict[str, Any] = None, 
                    user_id: Optional[int] = None, service_name: str = "",
                    severity: AlertLevel = AlertLevel.ERROR):
        """Record an error with context information"""
        error_info = ErrorInfo(
            error_type=type(error).__name__,
            error_message=str(error),
            traceback_str=traceback.format_exc(),
            context=context or {},
            user_id=user_id,
            service_name=service_name,
            severity=severity
        )
        
        with self._lock:
            self.errors.append(error_info)
            self.error_patterns[error_info.error_type] += 1
        
        # Log the error
        log_level = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.ERROR: logging.ERROR,
            AlertLevel.CRITICAL: logging.CRITICAL
        }.get(severity, logging.ERROR)
        
        self.logger.log(log_level, f"[{service_name}] {error_info.error_type}: {error_info.error_message}")
        
        # Check for critical patterns
        self._check_error_patterns(error_info)
    
    def _check_error_patterns(self, error_info: ErrorInfo):
        """Check for concerning error patterns"""
        recent_errors = [e for e in self.errors if 
                        (datetime.utcnow() - e.timestamp).total_seconds() < 300]  # Last 5 minutes
        
        # Check for error spikes
        if len(recent_errors) > 10:
            self.logger.critical(f"Error spike detected: {len(recent_errors)} errors in last 5 minutes")
        
        # Check for repeated errors from same user
        if error_info.user_id:
            user_errors = [e for e in recent_errors if e.user_id == error_info.user_id]
            if len(user_errors) > 5:
                self.logger.warning(f"User {error_info.user_id} experiencing repeated errors")
    
    def get_error_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get error summary for the specified time period"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_errors = [e for e in self.errors if e.timestamp >= cutoff_time]
        
        if not recent_errors:
            return {"message": "No errors in the specified period"}
        
        error_types = defaultdict(int)
        services = defaultdict(int)
        severities = defaultdict(int)
        
        for error in recent_errors:
            error_types[error.error_type] += 1
            services[error.service_name] += 1
            severities[error.severity.value] += 1
        
        return {
            "period_hours": hours,
            "total_errors": len(recent_errors),
            "error_types": dict(error_types),
            "services": dict(services),
            "severities": dict(severities),
            "most_common_error": max(error_types.items(), key=lambda x: x[1])[0] if error_types else None
        }

class ServiceHealthMonitor:
    """Monitor health of various services"""
    
    def __init__(self):
        self.services: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def register_service(self, name: str, health_check: Callable[[], bool], 
                        check_interval: int = 60):
        """Register a service for health monitoring"""
        with self._lock:
            self.services[name] = {
                "health_check": health_check,
                "check_interval": check_interval,
                "last_check": None,
                "status": ServiceStatus.UNKNOWN,
                "last_error": None,
                "consecutive_failures": 0
            }
    
    def check_service_health(self, service_name: str) -> ServiceStatus:
        """Check health of a specific service"""
        if service_name not in self.services:
            return ServiceStatus.UNKNOWN
        
        service = self.services[service_name]
        now = datetime.utcnow()
        
        # Check if we need to run health check
        if (service["last_check"] is None or 
            (now - service["last_check"]).total_seconds() >= service["check_interval"]):
            
            try:
                is_healthy = service["health_check"]()
                service["last_check"] = now
                
                if is_healthy:
                    service["status"] = ServiceStatus.HEALTHY
                    service["consecutive_failures"] = 0
                    service["last_error"] = None
                else:
                    service["consecutive_failures"] += 1
                    if service["consecutive_failures"] >= 3:
                        service["status"] = ServiceStatus.UNHEALTHY
                    else:
                        service["status"] = ServiceStatus.DEGRADED
                
            except Exception as e:
                service["last_error"] = str(e)
                service["consecutive_failures"] += 1
                service["status"] = ServiceStatus.UNHEALTHY
                self.logger.error(f"Health check failed for {service_name}: {e}")
        
        return service["status"]
    
    def get_all_service_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all registered services"""
        status_report = {}
        
        for service_name in self.services:
            status = self.check_service_health(service_name)
            service = self.services[service_name]
            
            status_report[service_name] = {
                "status": status.value,
                "last_check": service["last_check"].isoformat() if service["last_check"] else None,
                "consecutive_failures": service["consecutive_failures"],
                "last_error": service["last_error"]
            }
        
        return status_report

def performance_monitor(service_name: str = "unknown"):
    """Decorator to monitor function performance"""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                # Record error if error tracker is available
                if hasattr(monitoring_system, 'error_tracker'):
                    monitoring_system.error_tracker.record_error(
                        e, context={"function": func.__name__}, service_name=service_name
                    )
                raise
            finally:
                duration_ms = (time.time() - start_time) * 1000
                # Record performance if monitor is available
                if hasattr(monitoring_system, 'performance_monitor'):
                    monitoring_system.performance_monitor.record_request(
                        duration_ms, success, service_name
                    )
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                if hasattr(monitoring_system, 'error_tracker'):
                    monitoring_system.error_tracker.record_error(
                        e, context={"function": func.__name__}, service_name=service_name
                    )
                raise
            finally:
                duration_ms = (time.time() - start_time) * 1000
                if hasattr(monitoring_system, 'performance_monitor'):
                    monitoring_system.performance_monitor.record_request(
                        duration_ms, success, service_name
                    )
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

class MonitoringSystem:
    """Central monitoring system"""
    
    def __init__(self):
        self.performance_monitor = PerformanceMonitor()
        self.error_tracker = ErrorTracker()
        self.health_monitor = ServiceHealthMonitor()
        self.logger = logging.getLogger(__name__)
        self._monitoring_task = None
    
    def start_monitoring(self, interval: int = 60):
        """Start background monitoring"""
        if self._monitoring_task is None:
            self._monitoring_task = asyncio.create_task(self._monitoring_loop(interval))
    
    async def _monitoring_loop(self, interval: int):
        """Background monitoring loop"""
        while True:
            try:
                # Collect current metrics
                metrics = self.performance_monitor.get_current_metrics()
                
                # Check for alerts
                if metrics.memory_percent > 90:
                    self.logger.warning(f"High memory usage: {metrics.memory_percent:.1f}%")
                
                if metrics.cpu_percent > 80:
                    self.logger.warning(f"High CPU usage: {metrics.cpu_percent:.1f}%")
                
                if metrics.error_rate > 10:
                    self.logger.error(f"High error rate: {metrics.error_rate:.1f}%")
                
                # Check service health
                self.health_monitor.get_all_service_status()
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(interval)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "performance": self.performance_monitor.get_current_metrics().to_dict(),
            "performance_summary": self.performance_monitor.get_metrics_summary(),
            "error_summary": self.error_tracker.get_error_summary(),
            "service_health": self.health_monitor.get_all_service_status()
        }
    
    def export_metrics(self, file_path: str):
        """Export metrics to file"""
        status = self.get_system_status()
        with open(file_path, 'w') as f:
            json.dump(status, f, indent=2)

# Global monitoring system instance
monitoring_system = MonitoringSystem()

# Convenience functions
def record_error(error: Exception, context: Dict[str, Any] = None, 
                user_id: Optional[int] = None, service_name: str = "",
                severity: AlertLevel = AlertLevel.ERROR):
    """Record an error in the global monitoring system"""
    monitoring_system.error_tracker.record_error(error, context, user_id, service_name, severity)

def get_system_status() -> Dict[str, Any]:
    """Get current system status"""
    return monitoring_system.get_system_status()

def register_service_health_check(name: str, health_check: Callable[[], bool], 
                                 check_interval: int = 60):
    """Register a service for health monitoring"""
    monitoring_system.health_monitor.register_service(name, health_check, check_interval)