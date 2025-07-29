"""Monitoring and metrics collection for the Telegram Trading Bot
Provides Prometheus metrics for autoscaling and observability"""

import time
import threading
from typing import Dict, Any
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from logger import logger

class BotMetrics:
    """Prometheus metrics collector for the trading bot"""
    
    def __init__(self):
        """Initialize metrics collectors"""
        # Message metrics
        self.telegram_messages_total = Counter(
            'telegram_messages_total',
            'Total number of Telegram messages processed',
            ['command', 'user_id']
        )
        
        self.telegram_message_duration = Histogram(
            'telegram_message_duration_seconds',
            'Time spent processing Telegram messages',
            ['command']
        )
        
        # User session metrics
        self.active_users = Gauge(
            'telegram_active_users',
            'Number of active users in the last 5 minutes'
        )
        
        self.concurrent_sessions = Gauge(
            'telegram_concurrent_sessions',
            'Number of concurrent user sessions'
        )
        
        # System metrics
        self.bot_uptime = Gauge(
            'bot_uptime_seconds',
            'Bot uptime in seconds'
        )
        
        self.bot_ready = Gauge(
            'bot_ready',
            'Whether the bot is ready to serve requests (1=ready, 0=not ready)'
        )
        
        # AI/Trading metrics
        self.ai_requests_total = Counter(
            'ai_requests_total',
            'Total number of AI API requests',
            ['provider', 'model']
        )
        
        self.trading_signals_total = Counter(
            'trading_signals_total',
            'Total number of trading signals generated',
            ['signal_type', 'symbol']
        )
        
        self.deep_learning_predictions = Counter(
            'deep_learning_predictions_total',
            'Total number of deep learning predictions',
            ['model_type', 'symbol']
        )
        
        # Error metrics
        self.errors_total = Counter(
            'errors_total',
            'Total number of errors',
            ['error_type', 'component']
        )
        
        # Performance metrics
        self.response_time = Histogram(
            'response_time_seconds',
            'Response time for various operations',
            ['operation']
        )
        
        # Internal tracking
        self.start_time = time.time()
        self.user_sessions: Dict[str, float] = {}
        self.session_lock = threading.Lock()
        
        logger.info("Bot metrics initialized")
    
    def record_message(self, command: str, user_id: str, duration: float = None):
        """Record a processed Telegram message"""
        self.telegram_messages_total.labels(command=command, user_id=user_id).inc()
        
        if duration is not None:
            self.telegram_message_duration.labels(command=command).observe(duration)
        
        # Update user session
        with self.session_lock:
            self.user_sessions[user_id] = time.time()
            self._update_session_metrics()
    
    def record_ai_request(self, provider: str, model: str):
        """Record an AI API request"""
        self.ai_requests_total.labels(provider=provider, model=model).inc()
    
    def record_trading_signal(self, signal_type: str, symbol: str):
        """Record a trading signal generation"""
        self.trading_signals_total.labels(signal_type=signal_type, symbol=symbol).inc()
    
    def record_prediction(self, model_type: str, symbol: str):
        """Record a deep learning prediction"""
        self.deep_learning_predictions.labels(model_type=model_type, symbol=symbol).inc()
    
    def record_error(self, error_type: str, component: str):
        """Record an error occurrence"""
        self.errors_total.labels(error_type=error_type, component=component).inc()
    
    def record_response_time(self, operation: str, duration: float):
        """Record response time for an operation"""
        self.response_time.labels(operation=operation).observe(duration)
    
    def set_ready_status(self, ready: bool):
        """Set the bot ready status"""
        self.bot_ready.set(1 if ready else 0)
    
    def _update_session_metrics(self):
        """Update session-related metrics"""
        current_time = time.time()
        
        # Remove sessions older than 5 minutes
        active_threshold = current_time - 300  # 5 minutes
        active_sessions = {
            user_id: last_seen 
            for user_id, last_seen in self.user_sessions.items() 
            if last_seen > active_threshold
        }
        
        self.user_sessions = active_sessions
        
        # Update metrics
        self.active_users.set(len(active_sessions))
        self.concurrent_sessions.set(len(self.user_sessions))
        self.bot_uptime.set(current_time - self.start_time)
    
    def cleanup_old_sessions(self):
        """Periodic cleanup of old user sessions"""
        with self.session_lock:
            self._update_session_metrics()
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics"""
        with self.session_lock:
            self._update_session_metrics()
            
        # Get metric values using proper Prometheus client methods
        try:
            # For Counter metrics, we need to sum all label combinations
            total_messages = sum(sample.value for sample in self.telegram_messages_total.collect()[0].samples)
            total_errors = sum(sample.value for sample in self.errors_total.collect()[0].samples)
            ready_status = self.bot_ready._value.get()
        except Exception:
            # Fallback values if metrics collection fails
            total_messages = 0
            total_errors = 0
            ready_status = 0
            
        return {
            "active_users": len(self.user_sessions),
            "uptime_seconds": time.time() - self.start_time,
            "total_messages": total_messages,
            "total_errors": total_errors,
            "ready": bool(ready_status)
        }

# Global metrics instance
metrics = BotMetrics()

def start_metrics_server(port: int = 9090):
    """Start the Prometheus metrics HTTP server"""
    try:
        start_http_server(port)
        logger.info(f"Metrics server started on port {port}")
    except Exception as e:
        logger.error(f"Failed to start metrics server: {e}")

def setup_periodic_cleanup():
    """Setup periodic cleanup of metrics"""
    def cleanup_worker():
        while True:
            time.sleep(60)  # Cleanup every minute
            metrics.cleanup_old_sessions()
    
    cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
    cleanup_thread.start()
    logger.info("Metrics cleanup worker started")

# Decorator for timing operations
def time_operation(operation_name: str):
    """Decorator to time operations and record metrics"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                metrics.record_response_time(operation_name, duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                metrics.record_response_time(operation_name, duration)
                metrics.record_error(type(e).__name__, operation_name)
                raise
        return wrapper
    return decorator