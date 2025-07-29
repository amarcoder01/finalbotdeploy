"""Secure Logging Module
Implements secure logging with data sanitization and audit trails
"""
import logging
import os
import json
import time
import hashlib
from typing import Dict, Any, Optional
from datetime import datetime
from logging.handlers import RotatingFileHandler
import re
from security_config import SecurityConfig

class SecurityEventType:
    """Security event types for audit logging"""
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    INVALID_INPUT = "invalid_input"
    INJECTION_ATTEMPT = "injection_attempt"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_ACCESS = "data_access"
    CONFIGURATION_CHANGE = "configuration_change"
    ERROR = "error"
    WARNING = "warning"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"

class SecureFormatter(logging.Formatter):
    """Custom formatter that sanitizes sensitive data"""
    
    # Patterns for sensitive data
    SENSITIVE_PATTERNS = [
        (re.compile(r'\b[A-Za-z0-9]{20,}\b'), '[TOKEN]'),  # API tokens
        (re.compile(r'\bsk-[A-Za-z0-9]{32,}\b'), '[API_KEY]'),  # OpenAI keys
        (re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'), '[CARD]'),  # Credit cards
        (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), '[EMAIL]'),  # Emails
        (re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'), '[IP]'),  # IP addresses
        (re.compile(r'password[\s]*[:=][\s]*[^\s]+', re.IGNORECASE), 'password=[REDACTED]'),
        (re.compile(r'token[\s]*[:=][\s]*[^\s]+', re.IGNORECASE), 'token=[REDACTED]'),
        (re.compile(r'key[\s]*[:=][\s]*[^\s]+', re.IGNORECASE), 'key=[REDACTED]'),
        (re.compile(r'secret[\s]*[:=][\s]*[^\s]+', re.IGNORECASE), 'secret=[REDACTED]'),
    ]
    
    def format(self, record):
        """Format log record with data sanitization"""
        # Get the original formatted message
        formatted = super().format(record)
        
        # Sanitize sensitive data
        for pattern, replacement in self.SENSITIVE_PATTERNS:
            formatted = pattern.sub(replacement, formatted)
        
        return formatted

class SecurityAuditLogger:
    """Security audit logger for tracking security events"""
    
    def __init__(self, log_dir: str = "logs"):
        """Initialize security audit logger"""
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up audit log file
        audit_file = os.path.join(log_dir, "security_audit.log")
        
        # Create audit logger
        self.audit_logger = logging.getLogger("security_audit")
        self.audit_logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in self.audit_logger.handlers[:]:
            self.audit_logger.removeHandler(handler)
        
        # Add rotating file handler
        handler = RotatingFileHandler(
            audit_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        
        # Set secure permissions
        os.chmod(audit_file, 0o600)
        
        # Use JSON formatter for structured logging
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        
        self.audit_logger.addHandler(handler)
        self.audit_logger.propagate = False
    
    def log_security_event(self, event_type: str, user_id: Optional[str] = None, 
                          details: Optional[Dict[str, Any]] = None, 
                          severity: str = "INFO"):
        """Log a security event"""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "severity": severity,
            "user_id": self._hash_user_id(user_id) if user_id else None,
            "details": self._sanitize_details(details or {}),
            "session_id": self._generate_session_id()
        }
        
        self.audit_logger.info(json.dumps(event))
    
    def _hash_user_id(self, user_id: str) -> str:
        """Hash user ID for privacy"""
        security_config = SecurityConfig()
        return hashlib.sha256(f"{user_id}{security_config.SECRET_KEY}".encode()).hexdigest()[:16]
    
    def _sanitize_details(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize sensitive data from details"""
        sanitized = {}
        
        for key, value in details.items():
            if isinstance(value, str):
                # Check if key suggests sensitive data
                if any(sensitive in key.lower() for sensitive in 
                      ['password', 'token', 'key', 'secret', 'api']):
                    sanitized[key] = '[REDACTED]'
                else:
                    # Apply pattern-based sanitization
                    sanitized_value = value
                    for pattern, replacement in SecureFormatter.SENSITIVE_PATTERNS:
                        sanitized_value = pattern.sub(replacement, sanitized_value)
                    sanitized[key] = sanitized_value
            elif isinstance(value, (int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_details(value)
            else:
                sanitized[key] = str(value)
        
        return sanitized
    
    def _generate_session_id(self) -> str:
        """Generate a session ID for log correlation"""
        return hashlib.md5(f"{time.time()}{os.getpid()}".encode()).hexdigest()[:8]

class SecureLogger:
    """Main secure logger class"""
    
    def __init__(self, name: str = "secure_bot", log_dir: str = "logs"):
        """Initialize secure logger"""
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Create main logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Add console handler with secure formatter
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = SecureFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # Add file handler with rotation
        log_file = os.path.join(log_dir, f"{name}.log")
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = SecureFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Set secure permissions on log file
        if os.path.exists(log_file):
            os.chmod(log_file, 0o600)
        
        # Initialize audit logger
        self.audit_logger = SecurityAuditLogger(log_dir)
        
        self.logger.propagate = False
    
    def info(self, message: str, user_id: Optional[str] = None, **kwargs):
        """Log info message"""
        self.logger.info(message, **kwargs)
        if user_id:
            self.audit_logger.log_security_event(
                SecurityEventType.DATA_ACCESS,
                user_id=user_id,
                details={"message": message}
            )
    
    def warning(self, message: str, user_id: Optional[str] = None, **kwargs):
        """Log warning message"""
        self.logger.warning(message, **kwargs)
        self.audit_logger.log_security_event(
            SecurityEventType.WARNING,
            user_id=user_id,
            details={"message": message},
            severity="WARNING"
        )
    
    def error(self, message: str, user_id: Optional[str] = None, **kwargs):
        """Log error message"""
        self.logger.error(message, **kwargs)
        self.audit_logger.log_security_event(
            SecurityEventType.ERROR,
            user_id=user_id,
            details={"message": message},
            severity="ERROR"
        )
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(message, **kwargs)
    
    def critical(self, message: str, user_id: Optional[str] = None, **kwargs):
        """Log critical message"""
        self.logger.critical(message, **kwargs)
        self.audit_logger.log_security_event(
            SecurityEventType.ERROR,
            user_id=user_id,
            details={"message": message},
            severity="CRITICAL"
        )
    
    def log_login_attempt(self, user_id: str, success: bool, ip_address: Optional[str] = None):
        """Log login attempt"""
        event_type = SecurityEventType.LOGIN_SUCCESS if success else SecurityEventType.LOGIN_FAILURE
        severity = "INFO" if success else "WARNING"
        
        details = {
            "success": success,
            "ip_address": ip_address or "unknown"
        }
        
        self.audit_logger.log_security_event(
            event_type,
            user_id=user_id,
            details=details,
            severity=severity
        )
        
        message = f"Login {'successful' if success else 'failed'} for user"
        if success:
            self.info(message)
        else:
            self.warning(message)
    
    def log_rate_limit_exceeded(self, user_id: str, endpoint: str, attempts: int):
        """Log rate limit exceeded"""
        details = {
            "endpoint": endpoint,
            "attempts": attempts
        }
        
        self.audit_logger.log_security_event(
            SecurityEventType.RATE_LIMIT_EXCEEDED,
            user_id=user_id,
            details=details,
            severity="WARNING"
        )
        
        self.warning(f"Rate limit exceeded for user on endpoint {endpoint}")
    
    def log_injection_attempt(self, user_id: str, input_data: str, attack_type: str):
        """Log injection attempt"""
        details = {
            "attack_type": attack_type,
            "input_sample": input_data[:100] + "..." if len(input_data) > 100 else input_data
        }
        
        self.audit_logger.log_security_event(
            SecurityEventType.INJECTION_ATTEMPT,
            user_id=user_id,
            details=details,
            severity="CRITICAL"
        )
        
        self.critical(f"Injection attempt detected: {attack_type}")
    
    def log_unauthorized_access(self, user_id: str, resource: str, action: str):
        """Log unauthorized access attempt"""
        details = {
            "resource": resource,
            "action": action
        }
        
        self.audit_logger.log_security_event(
            SecurityEventType.UNAUTHORIZED_ACCESS,
            user_id=user_id,
            details=details,
            severity="ERROR"
        )
        
        self.error(f"Unauthorized access attempt to {resource}")
    
    def log_configuration_change(self, user_id: str, setting: str, old_value: Any, new_value: Any):
        """Log configuration change"""
        details = {
            "setting": setting,
            "old_value": str(old_value),
            "new_value": str(new_value)
        }
        
        self.audit_logger.log_security_event(
            SecurityEventType.CONFIGURATION_CHANGE,
            user_id=user_id,
            details=details,
            severity="INFO"
        )
        
        self.info(f"Configuration changed: {setting}")
    
    def log_suspicious_activity(self, user_id: str, activity: str, details: Dict[str, Any]):
        """Log suspicious activity"""
        log_details = {
            "activity": activity,
            **details
        }
        
        self.audit_logger.log_security_event(
            SecurityEventType.SUSPICIOUS_ACTIVITY,
            user_id=user_id,
            details=log_details,
            severity="WARNING"
        )
        
        self.warning(f"Suspicious activity detected: {activity}")
    
    def log_system_event(self, event_type: str, message: str, user_id: Optional[str] = None):
        """Log system event"""
        details = {
            "event_type": event_type,
            "message": message
        }
        
        self.audit_logger.log_security_event(
            SecurityEventType.DATA_ACCESS,
            user_id=user_id or "system",
            details=details,
            severity="INFO"
        )
        
        self.info(f"System event [{event_type}]: {message}")
    
    def log_security_event(self, event_type: str, message: str, user_id: Optional[str] = None, severity: str = "INFO"):
        """Log security event"""
        details = {
            "event_type": event_type,
            "message": message
        }
        
        # Map severity to SecurityEventType
        if severity.upper() == "CRITICAL":
            sec_event_type = SecurityEventType.ERROR
        elif severity.upper() == "WARNING":
            sec_event_type = SecurityEventType.WARNING
        else:
            sec_event_type = SecurityEventType.DATA_ACCESS
        
        self.audit_logger.log_security_event(
            sec_event_type,
            user_id=user_id or "system",
            details=details,
            severity=severity.upper()
        )
        
        # Log with appropriate level
        if severity.upper() == "CRITICAL":
            self.critical(f"Security event [{event_type}]: {message}")
        elif severity.upper() == "WARNING":
            self.warning(f"Security event [{event_type}]: {message}")
        elif severity.upper() == "ERROR":
            self.error(f"Security event [{event_type}]: {message}")
        else:
            self.info(f"Security event [{event_type}]: {message}")

# Global secure logger instance
secure_logger = SecureLogger()