"""Rate Limiting and Access Control Module
Implements rate limiting, user authentication, and access control
"""
import time
import asyncio
from typing import Dict, Optional, Set, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from security_config import security_config, SecurityError
from logger import logger
import json
import os

class AccessLevel(Enum):
    """User access levels"""
    GUEST = "guest"
    USER = "user"
    PREMIUM = "premium"
    ADMIN = "admin"
    BANNED = "banned"

@dataclass
class RateLimitInfo:
    """Rate limit information for a user"""
    requests: deque
    last_request: float
    burst_count: int
    total_requests: int
    violations: int
    locked_until: Optional[float] = None

@dataclass
class UserSession:
    """User session information"""
    user_id: str
    access_level: AccessLevel
    created_at: float
    last_activity: float
    session_token: Optional[str] = None
    failed_attempts: int = 0
    locked_until: Optional[float] = None

class RateLimiter:
    """Advanced rate limiting and access control system"""
    
    def __init__(self):
        """Initialize rate limiter"""
        self.rate_limits: Dict[str, RateLimitInfo] = {}
        self.user_sessions: Dict[str, UserSession] = {}
        self.blocked_ips: Set[str] = set()
        self.admin_users: Set[str] = set()
        self.premium_users: Set[str] = set()
        self.banned_users: Set[str] = set()
        
        # Load persistent data
        self._load_persistent_data()
        
        # Cleanup task will be started when event loop is available
        self._cleanup_task_started = False
        
        logger.info("Rate limiter initialized")
    
    def start_cleanup_task(self):
        """Start the cleanup task if not already started"""
        if not self._cleanup_task_started:
            try:
                asyncio.create_task(self._cleanup_task())
                self._cleanup_task_started = True
                logger.info("Rate limiter cleanup task started")
            except RuntimeError:
                # No event loop running, task will be started later
                pass
    
    def _load_persistent_data(self):
        """Load persistent rate limiting data"""
        try:
            data_file = os.path.join(os.path.dirname(__file__), '.rate_limit_data.json')
            if os.path.exists(data_file):
                with open(data_file, 'r') as f:
                    data = json.load(f)
                    self.admin_users = set(data.get('admin_users', []))
                    self.premium_users = set(data.get('premium_users', []))
                    self.banned_users = set(data.get('banned_users', []))
                    self.blocked_ips = set(data.get('blocked_ips', []))
                    logger.info("Rate limiting data loaded")
        except Exception as e:
            logger.error(f"Error loading rate limiting data: {e}")
    
    def _save_persistent_data(self):
        """Save persistent rate limiting data"""
        try:
            data_file = os.path.join(os.path.dirname(__file__), '.rate_limit_data.json')
            data = {
                'admin_users': list(self.admin_users),
                'premium_users': list(self.premium_users),
                'banned_users': list(self.banned_users),
                'blocked_ips': list(self.blocked_ips)
            }
            with open(data_file, 'w') as f:
                json.dump(data, f)
            os.chmod(data_file, 0o600)
        except Exception as e:
            logger.error(f"Error saving rate limiting data: {e}")
    
    async def _cleanup_task(self):
        """Periodic cleanup of old data"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                current_time = time.time()
                
                # Clean up old rate limit data
                for user_id in list(self.rate_limits.keys()):
                    rate_info = self.rate_limits[user_id]
                    
                    # Remove old requests from deque
                    while (rate_info.requests and 
                           current_time - rate_info.requests[0] > security_config.RATE_LIMIT_WINDOW):
                        rate_info.requests.popleft()
                    
                    # Remove inactive users
                    if (current_time - rate_info.last_request > 3600 and 
                        not rate_info.requests):
                        del self.rate_limits[user_id]
                
                # Clean up expired sessions
                for user_id in list(self.user_sessions.keys()):
                    session = self.user_sessions[user_id]
                    if (current_time - session.last_activity > security_config.SESSION_TIMEOUT):
                        del self.user_sessions[user_id]
                
                logger.debug("Rate limiter cleanup completed")
                
            except Exception as e:
                logger.error(f"Error in rate limiter cleanup: {e}")
    
    def get_user_access_level(self, user_id: str) -> AccessLevel:
        """Get user access level"""
        if user_id in self.banned_users:
            return AccessLevel.BANNED
        elif user_id in self.admin_users:
            return AccessLevel.ADMIN
        elif user_id in self.premium_users:
            return AccessLevel.PREMIUM
        else:
            return AccessLevel.USER
    
    def set_user_access_level(self, user_id: str, access_level: AccessLevel, admin_user_id: str):
        """Set user access level (admin only)"""
        if admin_user_id not in self.admin_users:
            raise SecurityError("Only admins can modify user access levels")
        
        # Remove from all sets first
        self.admin_users.discard(user_id)
        self.premium_users.discard(user_id)
        self.banned_users.discard(user_id)
        
        # Add to appropriate set
        if access_level == AccessLevel.ADMIN:
            self.admin_users.add(user_id)
        elif access_level == AccessLevel.PREMIUM:
            self.premium_users.add(user_id)
        elif access_level == AccessLevel.BANNED:
            self.banned_users.add(user_id)
        
        self._save_persistent_data()
        logger.info(f"User {user_id} access level changed to {access_level.value} by {admin_user_id}")
    
    def check_rate_limit(self, user_id: str, endpoint: str = "default") -> Tuple[bool, Dict[str, any]]:
        """Check if user is within rate limits"""
        current_time = time.time()
        
        # Check if user is banned
        if user_id in self.banned_users:
            return False, {'reason': 'banned', 'retry_after': None}
        
        # Get or create rate limit info
        if user_id not in self.rate_limits:
            self.rate_limits[user_id] = RateLimitInfo(
                requests=deque(),
                last_request=current_time,
                burst_count=0,
                total_requests=0,
                violations=0
            )
        
        rate_info = self.rate_limits[user_id]
        
        # Check if user is temporarily locked
        if rate_info.locked_until and current_time < rate_info.locked_until:
            return False, {
                'reason': 'locked',
                'retry_after': int(rate_info.locked_until - current_time)
            }
        
        # Clear lock if expired
        if rate_info.locked_until and current_time >= rate_info.locked_until:
            rate_info.locked_until = None
            rate_info.violations = 0
        
        # Get rate limits based on user access level
        access_level = self.get_user_access_level(user_id)
        rate_limit, burst_limit = self._get_rate_limits_for_access_level(access_level)
        
        # Remove old requests from the window
        while (rate_info.requests and 
               current_time - rate_info.requests[0] > security_config.RATE_LIMIT_WINDOW):
            rate_info.requests.popleft()
        
        # Check burst limit
        time_since_last = current_time - rate_info.last_request
        if time_since_last < 1:  # Less than 1 second
            rate_info.burst_count += 1
            if rate_info.burst_count > burst_limit:
                rate_info.violations += 1
                self._handle_rate_limit_violation(user_id, rate_info)
                return False, {
                    'reason': 'burst_limit',
                    'retry_after': 1
                }
        else:
            rate_info.burst_count = 0
        
        # Check rate limit
        if len(rate_info.requests) >= rate_limit:
            rate_info.violations += 1
            self._handle_rate_limit_violation(user_id, rate_info)
            return False, {
                'reason': 'rate_limit',
                'retry_after': int(security_config.RATE_LIMIT_WINDOW - 
                                 (current_time - rate_info.requests[0]))
            }
        
        # Update rate limit info
        rate_info.requests.append(current_time)
        rate_info.last_request = current_time
        rate_info.total_requests += 1
        
        return True, {
            'remaining': rate_limit - len(rate_info.requests),
            'reset_time': int(current_time + security_config.RATE_LIMIT_WINDOW)
        }
    
    def _get_rate_limits_for_access_level(self, access_level: AccessLevel) -> Tuple[int, int]:
        """Get rate limits based on access level"""
        if access_level == AccessLevel.ADMIN:
            return 1000, 50  # Very high limits for admins
        elif access_level == AccessLevel.PREMIUM:
            return 200, 20   # Higher limits for premium users
        elif access_level == AccessLevel.USER:
            return 3, 10     # 3 requests per window, high burst for testing
        else:  # GUEST
            return 3, 10     # 3 requests per window, high burst for testing
    
    def _handle_rate_limit_violation(self, user_id: str, rate_info: RateLimitInfo):
        """Handle rate limit violations"""
        logger.warning(f"Rate limit violation for user {user_id}: {rate_info.violations} violations")
        
        # Progressive penalties
        if rate_info.violations >= 5:
            # Lock for 15 minutes after 5 violations
            rate_info.locked_until = time.time() + security_config.LOCKOUT_DURATION
            logger.warning(f"User {user_id} locked for {security_config.LOCKOUT_DURATION} seconds")
        elif rate_info.violations >= 10:
            # Temporary ban for repeated violations
            self.banned_users.add(user_id)
            self._save_persistent_data()
            logger.warning(f"User {user_id} temporarily banned for repeated violations")
    
    def create_session(self, user_id: str) -> str:
        """Create a new user session"""
        current_time = time.time()
        access_level = self.get_user_access_level(user_id)
        
        session = UserSession(
            user_id=user_id,
            access_level=access_level,
            created_at=current_time,
            last_activity=current_time
        )
        
        # Generate session token
        session.session_token = security_config.generate_session_token(user_id)
        
        self.user_sessions[user_id] = session
        logger.info(f"Session created for user {user_id}")
        
        return session.session_token
    
    def validate_session(self, session_token: str) -> Optional[UserSession]:
        """Validate session token"""
        token_data = security_config.verify_session_token(session_token)
        if not token_data:
            return None
        
        user_id = token_data['user_id']
        if user_id not in self.user_sessions:
            return None
        
        session = self.user_sessions[user_id]
        if session.session_token != session_token:
            return None
        
        # Update last activity
        session.last_activity = time.time()
        
        return session
    
    def invalidate_session(self, user_id: str):
        """Invalidate user session"""
        if user_id in self.user_sessions:
            del self.user_sessions[user_id]
            logger.info(f"Session invalidated for user {user_id}")
    
    def record_failed_login(self, user_id: str) -> bool:
        """Record failed login attempt"""
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = UserSession(
                user_id=user_id,
                access_level=self.get_user_access_level(user_id),
                created_at=time.time(),
                last_activity=time.time()
            )
        
        session = self.user_sessions[user_id]
        session.failed_attempts += 1
        
        if session.failed_attempts >= security_config.MAX_LOGIN_ATTEMPTS:
            session.locked_until = time.time() + security_config.LOCKOUT_DURATION
            logger.warning(f"User {user_id} locked due to failed login attempts")
            return False
        
        return True
    
    def reset_failed_attempts(self, user_id: str):
        """Reset failed login attempts after successful login"""
        if user_id in self.user_sessions:
            self.user_sessions[user_id].failed_attempts = 0
            self.user_sessions[user_id].locked_until = None
    
    def get_rate_limit_status(self, user_id: str) -> Dict[str, any]:
        """Get current rate limit status for user"""
        if user_id not in self.rate_limits:
            return {
                'requests_made': 0,
                'requests_remaining': security_config.RATE_LIMIT_REQUESTS,
                'reset_time': int(time.time() + security_config.RATE_LIMIT_WINDOW),
                'violations': 0
            }
        
        rate_info = self.rate_limits[user_id]
        access_level = self.get_user_access_level(user_id)
        rate_limit, _ = self._get_rate_limits_for_access_level(access_level)
        
        return {
            'requests_made': len(rate_info.requests),
            'requests_remaining': max(0, rate_limit - len(rate_info.requests)),
            'reset_time': int(time.time() + security_config.RATE_LIMIT_WINDOW),
            'violations': rate_info.violations,
            'locked_until': rate_info.locked_until
        }

# Global rate limiter instance - will be initialized when needed
rate_limiter = None

def get_rate_limiter():
    """Get or create the global rate limiter instance"""
    global rate_limiter
    if rate_limiter is None:
        rate_limiter = RateLimiter()
    return rate_limiter