"""
Logging configuration for the Telegram Trading Bot
Provides structured logging for debugging and monitoring
"""
import logging
import sys
from datetime import datetime

class BotLogger:
    """Custom logger for the trading bot"""
    
    def __init__(self, name="TradingBot", level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup console and file handlers for logging"""
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Ensure logs directory exists
        import os
        logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
        if not os.path.exists(logs_dir):
            try:
                os.makedirs(logs_dir)
                print(f"Created logs directory at {logs_dir}")
            except Exception as e:
                print(f"Could not create logs directory: {e}")
        
        # File handler for info and above
        try:
            info_file_handler = logging.FileHandler(os.path.join(logs_dir, 'bot_info.log'), encoding='utf-8')
            info_file_handler.setLevel(logging.INFO)
            info_file_handler.setFormatter(formatter)
            self.logger.addHandler(info_file_handler)
        except Exception as e:
            self.logger.warning(f"Could not create info file handler: {e}")
        # File handler for errors
        try:
            file_handler = logging.FileHandler(os.path.join(logs_dir, 'bot_errors.log'), encoding='utf-8')
            file_handler.setLevel(logging.ERROR)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        except Exception as e:
            self.logger.warning(f"Could not create file handler: {e}")
    
    def info(self, message):
        """Log info message"""
        self.logger.info(message)
    
    def error(self, message):
        """Log error message"""
        self.logger.error(message)
    
    def warning(self, message):
        """Log warning message"""
        self.logger.warning(message)
    
    def debug(self, message):
        """Log debug message"""
        self.logger.debug(message)

# Global logger instance
logger = BotLogger()
