"""Configuration module for the Telegram Trading Bot
Handles environment variables and API configurations with security enhancements
"""
import os
import sys
from dotenv import load_dotenv
from security_config import SecurityConfig, SecurityError

# Load environment variables from .env file if it exists
# This will automatically search for .env files in the current and parent directories
load_dotenv()

class Config:
    """Configuration class for bot settings with security enhancements"""
    
    def __init__(self):
        # Initialize security configuration
        self.security = SecurityConfig()
        
    # Telegram Bot Configuration (encrypted storage)
    @property
    def TELEGRAM_BOT_TOKEN(self):
        token = os.getenv("TELEGRAM_API_TOKEN", "")
        if token and self.security.is_initialized():
            try:
                # Store encrypted version for future use
                encrypted_token = self.security.encrypt_sensitive_data(token)
                return token  # Return plain for immediate use
            except SecurityError:
                return token  # Fallback to plain if encryption fails
        return token
    
    # OpenAI Configuration (encrypted storage)
    @property
    def OPENAI_API_KEY(self):
        key = os.getenv("OPENAI_API_KEY", "")
        if key and self.security.is_initialized():
            try:
                # Store encrypted version for future use
                encrypted_key = self.security.encrypt_sensitive_data(key)
                return key  # Return plain for immediate use
            except SecurityError:
                return key  # Fallback to plain if encryption fails
        return key
    
    OPENAI_MODEL = "gpt-4o-mini"  # Using GPT-4o mini as specified
    
    # Alpaca Trading Configuration (encrypted storage)
    @property
    def ALPACA_API_KEY(self):
        key = os.getenv("ALPACA_API_KEY", "")
        if key and self.security.is_initialized():
            try:
                encrypted_key = self.security.encrypt_sensitive_data(key)
                return key
            except SecurityError:
                return key
        return key
    
    @property
    def ALPACA_API_SECRET(self):
        secret = os.getenv("ALPACA_API_SECRET", "")
        if secret and self.security.is_initialized():
            try:
                encrypted_secret = self.security.encrypt_sensitive_data(secret)
                return secret
            except SecurityError:
                return secret
        return secret
    
    ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    
    # Chart IMG API Configuration (encrypted storage)
    @property
    def CHART_IMG_API_KEY(self):
        key = os.getenv("CHART_IMG_API_KEY", "")
        if key and self.security.is_initialized():
            try:
                encrypted_key = self.security.encrypt_sensitive_data(key)
                return key
            except SecurityError:
                return key
        return key
    
    # Bot Configuration
    MAX_MESSAGE_LENGTH = 4000  # Telegram message limit
    REQUEST_TIMEOUT = 60  # Timeout for API requests (increased for complex analysis)
    
    # Security Configuration
    ENABLE_RATE_LIMITING = True
    ENABLE_INPUT_VALIDATION = True
    ENABLE_SECURE_LOGGING = True
    ENABLE_SESSION_MANAGEMENT = True
    
    @classmethod
    def validate_required_configs(cls):
        """Validate that all required configurations are present and initialize security"""
        # Create instance to access properties
        config_instance = cls()
        
        # Initialize security first
        try:
            config_instance.security.initialize_security()
            print("✅ Security configuration initialized")
        except SecurityError as e:
            print(f"⚠️ Security initialization warning: {e}")
            print("Continuing with basic security measures...")
        
        required_configs = {
            "TELEGRAM_BOT_TOKEN": config_instance.TELEGRAM_BOT_TOKEN,
            "OPENAI_API_KEY": config_instance.OPENAI_API_KEY,
        }
        
        missing_configs = []
        for config_name, config_value in required_configs.items():
            if not config_value:
                missing_configs.append(config_name)
        
        if missing_configs:
            print(f"❌ Missing required configurations: {', '.join(missing_configs)}")
            print("Please ensure all required environment variables are set.")
            sys.exit(1)
        else:
            print("✅ All required configurations are present")
            
        return config_instance
