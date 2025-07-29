"""Security Configuration Module
Centralized security settings and encryption utilities
"""
import os
import secrets
import hashlib
import hmac
import time
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import base64
from typing import Optional, Dict, Any
from logger import logger

class SecurityConfig:
    """Security configuration and encryption utilities"""
    
    def __init__(self):
        """Initialize security configuration"""
        self._encryption_key = None
        self._api_key_salt = None
        self._session_secret = None
        self._initialize_security_keys()
        
        # Security settings
        self.MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB
        self.MAX_MESSAGE_LENGTH = 4000
        self.MAX_SYMBOL_LENGTH = 10
        self.MAX_USERNAME_LENGTH = 50
        self.SESSION_TIMEOUT = 3600  # 1 hour
        self.MAX_LOGIN_ATTEMPTS = 5
        self.LOCKOUT_DURATION = 900  # 15 minutes
        
        # Rate limiting settings
        self.RATE_LIMIT_REQUESTS = 60  # requests per minute
        self.RATE_LIMIT_WINDOW = 60  # seconds
        self.BURST_LIMIT = 10  # burst requests
        
        # Input validation patterns
        self.ALLOWED_SYMBOLS_PATTERN = r'^[A-Z]{1,10}$'
        self.ALLOWED_COMMANDS_PATTERN = r'^/[a-zA-Z_]{1,20}$'
        self.NUMERIC_PATTERN = r'^[0-9]+\.?[0-9]*$'
        
        logger.info("Security configuration initialized")
    
    @property
    def SECRET_KEY(self) -> str:
        """Get the secret key for hashing operations"""
        return base64.urlsafe_b64encode(self._session_secret).decode()
    
    def _initialize_security_keys(self):
        """Initialize or load security keys"""
        try:
            # Try to load existing keys from secure storage
            key_file = os.path.join(os.path.dirname(__file__), '.security_keys')
            
            if os.path.exists(key_file):
                with open(key_file, 'rb') as f:
                    keys_data = f.read()
                    if len(keys_data) >= 96:  # 32 + 32 + 32 bytes
                        self._encryption_key = keys_data[:32]
                        self._api_key_salt = keys_data[32:64]
                        self._session_secret = keys_data[64:96]
                        logger.info("Security keys loaded from file")
                        return
            
            # Generate new keys if not found
            self._encryption_key = secrets.token_bytes(32)
            self._api_key_salt = secrets.token_bytes(32)
            self._session_secret = secrets.token_bytes(32)
            
            # Save keys securely
            with open(key_file, 'wb') as f:
                f.write(self._encryption_key + self._api_key_salt + self._session_secret)
            
            # Set restrictive permissions
            os.chmod(key_file, 0o600)
            logger.info("New security keys generated and saved")
            
        except Exception as e:
            logger.error(f"Error initializing security keys: {e}")
            # Fallback to environment-based keys
            self._encryption_key = hashlib.sha256(os.getenv('ENCRYPTION_SEED', 'fallback_key').encode()).digest()
            self._api_key_salt = hashlib.sha256(os.getenv('SALT_SEED', 'fallback_salt').encode()).digest()
            self._session_secret = hashlib.sha256(os.getenv('SESSION_SEED', 'fallback_session').encode()).digest()
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data using Fernet encryption"""
        try:
            fernet = Fernet(base64.urlsafe_b64encode(self._encryption_key))
            encrypted_data = fernet.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted_data).decode()
        except Exception as e:
            logger.error(f"Encryption error: {e}")
            raise SecurityError("Failed to encrypt data")
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        try:
            fernet = Fernet(base64.urlsafe_b64encode(self._encryption_key))
            decoded_data = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = fernet.decrypt(decoded_data)
            return decrypted_data.decode()
        except Exception as e:
            logger.error(f"Decryption error: {e}")
            raise SecurityError("Failed to decrypt data")
    
    def hash_api_key(self, api_key: str) -> str:
        """Hash API key for secure storage"""
        return hashlib.pbkdf2_hmac('sha256', api_key.encode(), self._api_key_salt, 100000).hex()
    
    def verify_api_key_hash(self, api_key: str, stored_hash: str) -> bool:
        """Verify API key against stored hash"""
        return hmac.compare_digest(self.hash_api_key(api_key), stored_hash)
    
    def generate_session_token(self, user_id: str) -> str:
        """Generate secure session token"""
        timestamp = str(int(time.time()))
        data = f"{user_id}:{timestamp}"
        signature = hmac.new(self._session_secret, data.encode(), hashlib.sha256).hexdigest()
        token = base64.urlsafe_b64encode(f"{data}:{signature}".encode()).decode()
        return token
    
    def verify_session_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode session token"""
        try:
            decoded = base64.urlsafe_b64decode(token.encode()).decode()
            parts = decoded.split(':')
            if len(parts) != 3:
                return None
            
            user_id, timestamp, signature = parts
            data = f"{user_id}:{timestamp}"
            expected_signature = hmac.new(self._session_secret, data.encode(), hashlib.sha256).hexdigest()
            
            if not hmac.compare_digest(signature, expected_signature):
                return None
            
            # Check if token is expired
            token_time = int(timestamp)
            if time.time() - token_time > self.SESSION_TIMEOUT:
                return None
            
            return {'user_id': user_id, 'timestamp': token_time}
        except Exception:
            return None
    
    def initialize_security(self) -> bool:
        """Initialize security system - for testing compatibility"""
        try:
            # Re-initialize security keys if needed
            if not all([self._encryption_key, self._api_key_salt, self._session_secret]):
                self._initialize_security_keys()
            logger.info("Security system initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize security system: {e}")
            return False
    
    def hash_user_id(self, user_id: str) -> str:
        """Hash user ID for secure storage and logging"""
        try:
            # Use HMAC with session secret for consistent hashing
            hashed = hmac.new(self._session_secret, str(user_id).encode(), hashlib.sha256).hexdigest()
            return hashed[:16]  # Return first 16 characters for brevity
        except Exception as e:
            logger.error(f"Error hashing user ID: {e}")
            return "unknown_user"
    
    def is_initialized(self) -> bool:
        """Check if security configuration is properly initialized"""
        return all([self._encryption_key, self._api_key_salt, self._session_secret])

class SecurityError(Exception):
    """Custom security exception"""
    pass

# Global security configuration instance
security_config = SecurityConfig()