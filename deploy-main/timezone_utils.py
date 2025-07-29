"""Modern timezone utilities using Python's built-in zoneinfo
Replaces legacy pytz with Python 3.9+ zoneinfo module
"""

import sys
from datetime import datetime, timezone, timedelta
from typing import Optional, Union
from logger import logger

# Use zoneinfo for Python 3.9+ or backports.zoneinfo for older versions
if sys.version_info >= (3, 9):
    from zoneinfo import ZoneInfo
    ZONEINFO_AVAILABLE = True
else:
    try:
        from backports.zoneinfo import ZoneInfo
        ZONEINFO_AVAILABLE = True
        logger.info("Using backports.zoneinfo for timezone support")
    except ImportError:
        ZONEINFO_AVAILABLE = False
        logger.warning("zoneinfo not available, using basic timezone support")
        
        # Fallback ZoneInfo class for basic timezone support
        class ZoneInfo:
            def __init__(self, key: str):
                self.key = key
                # Basic timezone mappings
                self._offset_map = {
                    'UTC': 0,
                    'US/Eastern': -5,
                    'US/Central': -6,
                    'US/Mountain': -7,
                    'US/Pacific': -8,
                    'Europe/London': 0,
                    'Europe/Berlin': 1,
                    'Asia/Tokyo': 9,
                    'Asia/Shanghai': 8,
                    'Australia/Sydney': 10,
                    'Asia/Kolkata': 5.5
                }
            
            def __str__(self):
                return self.key
            
            def utcoffset(self, dt):
                offset_hours = self._offset_map.get(self.key, 0)
                return timedelta(hours=offset_hours)

class ModernTimezoneHandler:
    """Modern timezone handling using zoneinfo"""
    
    # Common timezone mappings
    TIMEZONE_ALIASES = {
        'EST': 'US/Eastern',
        'CST': 'US/Central', 
        'MST': 'US/Mountain',
        'PST': 'US/Pacific',
        'GMT': 'UTC',
        'BST': 'Europe/London',
        'CET': 'Europe/Berlin',
        'JST': 'Asia/Tokyo',
        'CST_CHINA': 'Asia/Shanghai',
        'AEST': 'Australia/Sydney',
        'IST': 'Asia/Kolkata',
        'INDIA': 'Asia/Kolkata'
    }
    
    @classmethod
    def get_timezone(cls, tz_name: str) -> Optional[ZoneInfo]:
        """Get timezone object from name
        
        Args:
            tz_name: Timezone name (e.g., 'UTC', 'US/Eastern', 'EST')
            
        Returns:
            ZoneInfo object or None if invalid
        """
        try:
            # Handle aliases
            actual_tz_name = cls.TIMEZONE_ALIASES.get(tz_name, tz_name)
            
            if not ZONEINFO_AVAILABLE:
                logger.warning(f"Using fallback timezone for {actual_tz_name}")
            
            return ZoneInfo(actual_tz_name)
            
        except Exception as e:
            logger.error(f"Error creating timezone {tz_name}: {e}")
            return None
    
    @classmethod
    def utc_now(cls) -> datetime:
        """Get current UTC datetime"""
        return datetime.now(timezone.utc)
    
    @classmethod
    def now_in_timezone(cls, tz_name: str) -> Optional[datetime]:
        """Get current time in specified timezone
        
        Args:
            tz_name: Timezone name
            
        Returns:
            Current datetime in specified timezone or None if invalid
        """
        try:
            tz = cls.get_timezone(tz_name)
            if tz is None:
                return None
            
            return datetime.now(tz)
            
        except Exception as e:
            logger.error(f"Error getting time in timezone {tz_name}: {e}")
            return None
    
    @classmethod
    def convert_timezone(cls, dt: datetime, from_tz: str, to_tz: str) -> Optional[datetime]:
        """Convert datetime from one timezone to another
        
        Args:
            dt: Datetime object (naive or aware)
            from_tz: Source timezone name
            to_tz: Target timezone name
            
        Returns:
            Converted datetime or None if error
        """
        try:
            from_timezone = cls.get_timezone(from_tz)
            to_timezone = cls.get_timezone(to_tz)
            
            if from_timezone is None or to_timezone is None:
                return None
            
            # If datetime is naive, localize it to from_timezone
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=from_timezone)
            
            # Convert to target timezone
            return dt.astimezone(to_timezone)
            
        except Exception as e:
            logger.error(f"Error converting timezone from {from_tz} to {to_tz}: {e}")
            return None
    
    @classmethod
    def localize_datetime(cls, dt: datetime, tz_name: str) -> Optional[datetime]:
        """Add timezone info to naive datetime
        
        Args:
            dt: Naive datetime object
            tz_name: Timezone name
            
        Returns:
            Timezone-aware datetime or None if error
        """
        try:
            if dt.tzinfo is not None:
                logger.warning("Datetime already has timezone info")
                return dt
            
            tz = cls.get_timezone(tz_name)
            if tz is None:
                return None
            
            return dt.replace(tzinfo=tz)
            
        except Exception as e:
            logger.error(f"Error localizing datetime to {tz_name}: {e}")
            return None
    
    @classmethod
    def to_utc(cls, dt: datetime, source_tz: Optional[str] = None) -> Optional[datetime]:
        """Convert datetime to UTC
        
        Args:
            dt: Datetime object
            source_tz: Source timezone if dt is naive
            
        Returns:
            UTC datetime or None if error
        """
        try:
            # If datetime is naive and source_tz provided, localize first
            if dt.tzinfo is None and source_tz:
                dt = cls.localize_datetime(dt, source_tz)
                if dt is None:
                    return None
            
            # If still naive, assume UTC
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            
            return dt.astimezone(timezone.utc)
            
        except Exception as e:
            logger.error(f"Error converting to UTC: {e}")
            return None
    
    @classmethod
    def format_datetime(cls, dt: datetime, fmt: str = '%Y-%m-%d %H:%M:%S %Z') -> str:
        """Format datetime with timezone info
        
        Args:
            dt: Datetime object
            fmt: Format string
            
        Returns:
            Formatted datetime string
        """
        try:
            return dt.strftime(fmt)
        except Exception as e:
            logger.error(f"Error formatting datetime: {e}")
            return str(dt)
    
    @classmethod
    def parse_datetime(cls, dt_str: str, fmt: str, tz_name: Optional[str] = None) -> Optional[datetime]:
        """Parse datetime string with optional timezone
        
        Args:
            dt_str: Datetime string
            fmt: Format string
            tz_name: Timezone name for naive datetimes
            
        Returns:
            Parsed datetime or None if error
        """
        try:
            dt = datetime.strptime(dt_str, fmt)
            
            # If timezone name provided and datetime is naive, localize
            if tz_name and dt.tzinfo is None:
                dt = cls.localize_datetime(dt, tz_name)
            
            return dt
            
        except Exception as e:
            logger.error(f"Error parsing datetime '{dt_str}' with format '{fmt}': {e}")
            return None
    
    @classmethod
    def get_market_timezone(cls, market: str = 'US') -> Optional[ZoneInfo]:
        """Get timezone for financial markets
        
        Args:
            market: Market identifier ('US', 'EU', 'ASIA', etc.)
            
        Returns:
            Market timezone or None if unknown
        """
        market_timezones = {
            'US': 'US/Eastern',
            'NYSE': 'US/Eastern',
            'NASDAQ': 'US/Eastern',
            'EU': 'Europe/London',
            'LSE': 'Europe/London',
            'EURONEXT': 'Europe/Berlin',
            'ASIA': 'Asia/Tokyo',
            'TSE': 'Asia/Tokyo',
            'SSE': 'Asia/Shanghai',
            'ASX': 'Australia/Sydney'
        }
        
        tz_name = market_timezones.get(market.upper())
        if tz_name:
            return cls.get_timezone(tz_name)
        
        logger.warning(f"Unknown market timezone: {market}")
        return None
    
    @classmethod
    def is_market_open(cls, market: str = 'US', dt: Optional[datetime] = None) -> bool:
        """Check if market is currently open (basic implementation)
        
        Args:
            market: Market identifier
            dt: Datetime to check (defaults to now)
            
        Returns:
            True if market is likely open
        """
        try:
            if dt is None:
                dt = cls.utc_now()
            
            market_tz = cls.get_market_timezone(market)
            if market_tz is None:
                return False
            
            # Convert to market timezone
            market_time = dt.astimezone(market_tz)
            
            # Basic market hours (simplified)
            weekday = market_time.weekday()  # 0=Monday, 6=Sunday
            hour = market_time.hour
            
            # Weekend check
            if weekday >= 5:  # Saturday or Sunday
                return False
            
            # Basic market hours (9:30 AM - 4:00 PM for US markets)
            if market.upper() in ['US', 'NYSE', 'NASDAQ']:
                return 9 <= hour < 16  # Simplified, doesn't account for 9:30 start
            
            # Default business hours
            return 9 <= hour < 17
            
        except Exception as e:
            logger.error(f"Error checking market hours for {market}: {e}")
            return False

# Convenience functions for backward compatibility
def utc_now() -> datetime:
    """Get current UTC datetime"""
    return ModernTimezoneHandler.utc_now()

def get_timezone(tz_name: str) -> Optional[ZoneInfo]:
    """Get timezone object"""
    return ModernTimezoneHandler.get_timezone(tz_name)

def convert_timezone(dt: datetime, from_tz: str, to_tz: str) -> Optional[datetime]:
    """Convert between timezones"""
    return ModernTimezoneHandler.convert_timezone(dt, from_tz, to_tz)

def to_utc(dt: datetime, source_tz: Optional[str] = None) -> Optional[datetime]:
    """Convert to UTC"""
    return ModernTimezoneHandler.to_utc(dt, source_tz)

def get_ist_time() -> Optional[datetime]:
    """Get current time in IST (Asia/Kolkata)"""
    return ModernTimezoneHandler.now_in_timezone('Asia/Kolkata')

def format_ist_timestamp(fmt: str = '%Y-%m-%d %H:%M:%S') -> str:
    """Get formatted IST timestamp"""
    ist_time = get_ist_time()
    if ist_time:
        return ist_time.strftime(fmt)
    else:
        # Fallback to UTC with manual IST conversion
        utc_time = utc_now()
        ist_time = utc_time + timedelta(hours=5, minutes=30)
        return ist_time.strftime(fmt) + ' IST'

def convert_to_ist(dt: datetime, source_tz: Optional[str] = None) -> Optional[datetime]:
    """Convert any datetime to IST"""
    return ModernTimezoneHandler.convert_timezone(dt, source_tz or 'UTC', 'Asia/Kolkata')