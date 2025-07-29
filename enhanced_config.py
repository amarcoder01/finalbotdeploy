#!/usr/bin/env python3
"""
Enhanced Configuration Management
Provides robust configuration handling with validation, environment detection, and security
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from dotenv import load_dotenv

class Environment(Enum):
    """Application environment types"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    LOCAL = "local"

@dataclass
class APIConfig:
    """Configuration for external APIs"""
    name: str
    api_key: str
    base_url: str = ""
    timeout: int = 30
    rate_limit: int = 100
    retry_attempts: int = 3
    required: bool = True
    
    def is_valid(self) -> bool:
        """Check if API configuration is valid"""
        if self.required and not self.api_key:
            return False
        if self.api_key and len(self.api_key) < 10:
            return False
        return True
    
    def mask_key(self) -> str:
        """Return masked API key for logging"""
        if not self.api_key:
            return "[NOT_SET]"
        if len(self.api_key) < 8:
            return "[INVALID]"
        return f"{self.api_key[:4]}...{self.api_key[-4:]}"

@dataclass
class DatabaseConfig:
    """Database configuration"""
    url: str = ""
    max_connections: int = 10
    timeout: int = 30
    pool_size: int = 5
    echo: bool = False
    
    def is_configured(self) -> bool:
        """Check if database is configured"""
        return bool(self.url)

@dataclass
class SecurityConfig:
    """Security-related configuration"""
    allowed_user_ids: List[int] = field(default_factory=list)
    admin_user_ids: List[int] = field(default_factory=list)
    rate_limit_per_user: int = 60
    max_message_length: int = 4000
    enable_logging: bool = True
    log_user_messages: bool = False  # Privacy setting
    
@dataclass
class PerformanceConfig:
    """Performance-related configuration"""
    request_timeout: int = 30
    max_concurrent_requests: int = 10
    cache_ttl: int = 300
    enable_caching: bool = True
    max_memory_mb: int = 512
    cleanup_interval: int = 3600

class EnhancedConfig:
    """Enhanced configuration management with validation and environment detection"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.environment = self._detect_environment()
        self._load_environment_files()
        self._initialize_configs()
        self._validate_configuration()
    
    def _detect_environment(self) -> Environment:
        """Detect current environment"""
        env_name = os.getenv("ENVIRONMENT", "development").lower()
        try:
            return Environment(env_name)
        except ValueError:
            return Environment.DEVELOPMENT
    
    def _load_environment_files(self):
        """Load environment files in order of precedence"""
        # Load base .env file
        base_env = Path(__file__).parent / ".env"
        if base_env.exists():
            load_dotenv(base_env)
        
        # Load environment-specific .env file
        env_specific = Path(__file__).parent / f".env.{self.environment.value}"
        if env_specific.exists():
            load_dotenv(env_specific, override=True)
        
        # Load local .env file (highest precedence)
        local_env = Path(__file__).parent / ".env.local"
        if local_env.exists():
            load_dotenv(local_env, override=True)
    
    def _initialize_configs(self):
        """Initialize all configuration sections"""
        # Core application settings
        self.APP_NAME = os.getenv("APP_NAME", "AI Trading Bot")
        self.APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
        self.DEBUG = os.getenv("DEBUG", "false").lower() == "true"
        
        # API Configurations
        self.apis = {
            "telegram": APIConfig(
                name="Telegram Bot API",
                api_key=os.getenv("TELEGRAM_API_TOKEN", ""),
                base_url="https://api.telegram.org",
                required=True
            ),
            "openai": APIConfig(
                name="OpenAI API",
                api_key=os.getenv("OPENAI_API_KEY", ""),
                base_url="https://api.openai.com",
                required=True
            ),
            "alpaca": APIConfig(
                name="Alpaca Trading API",
                api_key=os.getenv("ALPACA_API_KEY", ""),
                base_url=os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
                required=False
            ),
            "alpha_vantage": APIConfig(
                name="Alpha Vantage API",
                api_key=os.getenv("ALPHA_VANTAGE_API_KEY", ""),
                base_url="https://www.alphavantage.co",
                required=False
            )
        }
        
        # Additional Alpaca secret
        self.ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET", "")
        
        # OpenAI Model Configuration
        self.OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "1000"))
        self.OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
        
        # Database Configuration
        self.database = DatabaseConfig(
            url=os.getenv("DATABASE_URL", ""),
            max_connections=int(os.getenv("DB_MAX_CONNECTIONS", "10")),
            timeout=int(os.getenv("DB_TIMEOUT", "30")),
            pool_size=int(os.getenv("DB_POOL_SIZE", "5")),
            echo=os.getenv("DB_ECHO", "false").lower() == "true"
        )
        
        # Security Configuration
        allowed_users = os.getenv("ALLOWED_USER_IDS", "")
        admin_users = os.getenv("ADMIN_USER_IDS", "")
        
        self.security = SecurityConfig(
            allowed_user_ids=self._parse_user_ids(allowed_users),
            admin_user_ids=self._parse_user_ids(admin_users),
            rate_limit_per_user=int(os.getenv("RATE_LIMIT_PER_USER", "60")),
            max_message_length=int(os.getenv("MAX_MESSAGE_LENGTH", "4000")),
            enable_logging=os.getenv("ENABLE_LOGGING", "true").lower() == "true",
            log_user_messages=os.getenv("LOG_USER_MESSAGES", "false").lower() == "true"
        )
        
        # Performance Configuration
        self.performance = PerformanceConfig(
            request_timeout=int(os.getenv("REQUEST_TIMEOUT", "30")),
            max_concurrent_requests=int(os.getenv("MAX_CONCURRENT_REQUESTS", "10")),
            cache_ttl=int(os.getenv("CACHE_TTL", "300")),
            enable_caching=os.getenv("ENABLE_CACHING", "true").lower() == "true",
            max_memory_mb=int(os.getenv("MAX_MEMORY_MB", "512")),
            cleanup_interval=int(os.getenv("CLEANUP_INTERVAL", "3600"))
        )
        
        # Feature Flags
        self.features = {
            "enable_qlib": os.getenv("ENABLE_QLIB", "true").lower() == "true",
            "enable_deep_learning": os.getenv("ENABLE_DEEP_LEARNING", "true").lower() == "true",
            "enable_backtesting": os.getenv("ENABLE_BACKTESTING", "true").lower() == "true",
            "enable_alerts": os.getenv("ENABLE_ALERTS", "true").lower() == "true",
            "enable_portfolio_tracking": os.getenv("ENABLE_PORTFOLIO_TRACKING", "true").lower() == "true",
            "enable_web_search": os.getenv("ENABLE_WEB_SEARCH", "true").lower() == "true"
        }
    
    def _parse_user_ids(self, user_ids_str: str) -> List[int]:
        """Parse comma-separated user IDs"""
        if not user_ids_str:
            return []
        try:
            return [int(uid.strip()) for uid in user_ids_str.split(",") if uid.strip()]
        except ValueError:
            return []
    
    def _validate_configuration(self):
        """Validate all configuration settings"""
        errors = []
        warnings = []
        
        # Validate required APIs
        for api_name, api_config in self.apis.items():
            if not api_config.is_valid():
                if api_config.required:
                    errors.append(f"Required API '{api_name}' is not properly configured")
                else:
                    warnings.append(f"Optional API '{api_name}' is not configured")
        
        # Validate OpenAI settings
        if self.OPENAI_MAX_TOKENS < 100 or self.OPENAI_MAX_TOKENS > 4000:
            warnings.append("OPENAI_MAX_TOKENS should be between 100 and 4000")
        
        if self.OPENAI_TEMPERATURE < 0 or self.OPENAI_TEMPERATURE > 2:
            warnings.append("OPENAI_TEMPERATURE should be between 0 and 2")
        
        # Validate performance settings
        if self.performance.request_timeout < 5:
            warnings.append("REQUEST_TIMEOUT is very low, may cause timeouts")
        
        if self.performance.max_memory_mb < 256:
            warnings.append("MAX_MEMORY_MB is very low, may cause performance issues")
        
        # Log validation results
        logger = logging.getLogger(__name__)
        
        if errors:
            for error in errors:
                logger.error(f"Configuration Error: {error}")
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
        
        if warnings:
            for warning in warnings:
                logger.warning(f"Configuration Warning: {warning}")
    
    def get_api_config(self, api_name: str) -> Optional[APIConfig]:
        """Get API configuration by name"""
        return self.apis.get(api_name)
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a feature is enabled"""
        return self.features.get(feature_name, False)
    
    def is_user_allowed(self, user_id: int) -> bool:
        """Check if user is allowed to use the bot"""
        if not self.security.allowed_user_ids:
            return True  # No restrictions if list is empty
        return user_id in self.security.allowed_user_ids
    
    def is_admin_user(self, user_id: int) -> bool:
        """Check if user is an admin"""
        return user_id in self.security.admin_user_ids
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for debugging"""
        return {
            "environment": "local",
            "app_name": self.APP_NAME,
            "app_version": self.APP_VERSION,
            "debug": self.DEBUG,
            "apis": {
                name: {
                    "configured": config.is_valid(),
                    "required": config.required,
                    "api_key": config.mask_key()
                }
                for name, config in self.apis.items()
            },
            "database_configured": self.database.is_configured(),
            "features": self.features,
            "security": {
                "allowed_users_count": len(self.security.allowed_user_ids),
                "admin_users_count": len(self.security.admin_user_ids),
                "rate_limit": self.security.rate_limit_per_user
            },
            "performance": {
                "timeout": self.performance.request_timeout,
                "caching_enabled": self.performance.enable_caching,
                "max_memory_mb": self.performance.max_memory_mb
            }
        }
    
    def export_config(self, file_path: str, include_secrets: bool = False):
        """Export configuration to file"""
        config_data = self.get_config_summary()
        
        if not include_secrets:
            # Remove sensitive information
            if "apis" in config_data:
                for api_config in config_data["apis"].values():
                    api_config.pop("api_key", None)
        
        with open(file_path, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    @classmethod
    def validate_required_configs(cls) -> bool:
        """Validate that all required configuration values are present"""
        try:
            config = cls()
            return True
        except ValueError:
            return False

# Global configuration instance
config = EnhancedConfig()

# Backward compatibility with existing Config class
class Config:
    """Backward compatibility wrapper for existing code"""
    
    @classmethod
    def __getattr__(cls, name: str):
        """Delegate attribute access to enhanced config"""
        # Map old attribute names to new config structure
        mapping = {
            "TELEGRAM_BOT_TOKEN": lambda: config.apis["telegram"].api_key,
            "OPENAI_API_KEY": lambda: config.apis["openai"].api_key,
            "ALPACA_API_KEY": lambda: config.apis["alpaca"].api_key,
            "ALPACA_API_SECRET": lambda: config.ALPACA_API_SECRET,
            "ALPACA_BASE_URL": lambda: config.apis["alpaca"].base_url,
            "OPENAI_MODEL": lambda: config.OPENAI_MODEL,
            "MAX_MESSAGE_LENGTH": lambda: config.security.max_message_length,
            "REQUEST_TIMEOUT": lambda: config.performance.request_timeout
        }
        
        if name in mapping:
            return mapping[name]()
        
        # Fallback to enhanced config attributes
        return getattr(config, name)
    
    @classmethod
    def validate_required_configs(cls) -> bool:
        """Validate required configurations"""
        return EnhancedConfig.validate_required_configs()