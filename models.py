from sqlalchemy import Column, Integer, String, DateTime, Boolean, Float, ForeignKey, Text
from sqlalchemy.orm import relationship
from db import Base
from datetime import datetime
from security_config import SecurityConfig
from rate_limiter import AccessLevel
import json

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    telegram_id = Column(String, unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Security and access control fields
    access_level = Column(String, default=AccessLevel.USER.value)
    is_active = Column(Boolean, default=True)
    failed_login_attempts = Column(Integer, default=0)
    last_login_attempt = Column(DateTime, nullable=True)
    last_successful_login = Column(DateTime, nullable=True)
    
    # Session management
    current_session_id = Column(String, nullable=True)
    session_expires_at = Column(DateTime, nullable=True)
    
    # Security tracking
    security_events_count = Column(Integer, default=0)
    last_security_event = Column(DateTime, nullable=True)
    
    # Encrypted sensitive data (JSON field for flexibility)
    encrypted_data = Column(Text, nullable=True)
    
    alerts = relationship("Alert", back_populates="user")
    trades = relationship("Trade", back_populates="user")
    preferences = relationship("UserPreference", back_populates="user", cascade="all, delete-orphan")
    security_logs = relationship("SecurityLog", back_populates="user", cascade="all, delete-orphan")
    watchlist = relationship("Watchlist", back_populates="user", cascade="all, delete-orphan")
    
    def get_access_level(self):
        """Get user's access level as enum"""
        try:
            return AccessLevel(self.access_level)
        except ValueError:
            return AccessLevel.USER
    
    def set_access_level(self, level: AccessLevel):
        """Set user's access level"""
        self.access_level = level.value
    
    def store_encrypted_data(self, key: str, value: str, security_config: SecurityConfig):
        """Store encrypted sensitive data"""
        if not self.encrypted_data:
            self.encrypted_data = '{}'
        
        data = json.loads(self.encrypted_data)
        encrypted_value = security_config.encrypt_sensitive_data(value)
        data[key] = encrypted_value
        self.encrypted_data = json.dumps(data)
    
    def get_encrypted_data(self, key: str, security_config: SecurityConfig):
        """Retrieve and decrypt sensitive data"""
        if not self.encrypted_data:
            return None
        
        data = json.loads(self.encrypted_data)
        encrypted_value = data.get(key)
        if encrypted_value:
            return security_config.decrypt_sensitive_data(encrypted_value)
        return None

class Alert(Base):
    __tablename__ = "alerts"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    symbol = Column(String, nullable=False)
    condition = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Security fields
    created_from_ip = Column(String, nullable=True)
    last_triggered = Column(DateTime, nullable=True)
    trigger_count = Column(Integer, default=0)
    
    user = relationship("User", back_populates="alerts")

class Trade(Base):
    __tablename__ = "trades"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    symbol = Column(String, nullable=False)
    action = Column(String, nullable=False)  # buy/sell
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    executed_at = Column(DateTime, default=datetime.utcnow)
    
    # Security and audit fields
    created_from_ip = Column(String, nullable=True)
    validation_status = Column(String, default="pending")  # pending, validated, rejected
    risk_score = Column(Float, default=0.0)
    
    user = relationship("User", back_populates="trades")

class UserPreference(Base):
    __tablename__ = "user_preferences"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    key = Column(String, nullable=False)
    value = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="preferences")

class Watchlist(Base):
    __tablename__ = "watchlist"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    symbol = Column(String, nullable=False)
    added_at = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text, nullable=True)
    
    # Security fields
    created_from_ip = Column(String, nullable=True)
    
    user = relationship("User", back_populates="watchlist")

class SecurityLog(Base):
    __tablename__ = "security_logs"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    event_type = Column(String, nullable=False)
    event_details = Column(Text, nullable=True)
    ip_address = Column(String, nullable=True)
    user_agent = Column(String, nullable=True)
    severity = Column(String, default="info")  # info, warning, error, critical
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Additional security context
    session_id = Column(String, nullable=True)
    endpoint = Column(String, nullable=True)
    success = Column(Boolean, default=True)
    error_message = Column(Text, nullable=True)
    
    user = relationship("User", back_populates="security_logs")
    
    def to_dict(self):
        """Convert security log to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'event_type': self.event_type,
            'event_details': self.event_details,
            'ip_address': self.ip_address,
            'severity': self.severity,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'session_id': self.session_id,
            'endpoint': self.endpoint,
            'success': self.success,
            'error_message': self.error_message
        }