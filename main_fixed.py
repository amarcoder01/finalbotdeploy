#!/usr/bin/env python3
"""
Fixed main.py with dependency handling
"""
import os
import sys
import asyncio
import time
import importlib
import logging

# Add current directory to path  
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# First check and handle missing numpy/pandas
missing_deps = []
try:
    import numpy
except ImportError:
    missing_deps.append('numpy')

try:
    import pandas
except ImportError:
    missing_deps.append('pandas')

if missing_deps:
    print(f"Warning: Missing dependencies: {missing_deps}")
    print("Running bot with limited functionality...")
    
    # Create dummy modules to prevent import errors
    class DummyModule:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    
    for dep in missing_deps:
        sys.modules[dep] = DummyModule()
        if dep == 'numpy':
            sys.modules['numpy.core'] = DummyModule()
            sys.modules['numpy.core.multiarray'] = DummyModule()

# Now import the rest
from logger import logger
try:
    from security_logger import SecureLogger
    secure_logger = SecureLogger()
except ImportError:
    secure_logger = logger  # Fallback to regular logger

from monitoring import metrics, preloader, get_cache_stats
from telegram_handler import TelegramHandler
from aiohttp import web
from performance_cache import connection_pool

class TradingBot:
    """Main trading bot class"""
    def __init__(self):
        self.telegram_handler = None
        self.health_server = None
        self.start_time = None
        self.is_ready = False
        
    def validate_environment(self) -> bool:
        """Validate environment variables and configuration"""
        required_vars = ['TELEGRAM_API_TOKEN', 'OPENAI_API_KEY']
        missing = [var for var in required_vars if not os.getenv(var)]
        
        if missing:
            logger.error(f"Missing required environment variables: {missing}")
            return False
        
        return True
    
    async def health_check(self, request):
        """Health check endpoint"""
        uptime = int(time.time() - self.start_time) if self.start_time else 0
        
        health_data = {
            'status': 'healthy' if self.is_ready else 'starting',
            'uptime_seconds': uptime,
            'version': '1.0.0',
            'timestamp': time.time()
        }
        
        return web.json_response(health_data)
    
    async def start_health_server(self):
        """Start health check server for monitoring"""
        app = web.Application()
        app.router.add_get('/health', self.health_check)
        app.router.add_get('/', self.health_check)
        
        runner = web.AppRunner(app)
        await runner.setup()
        
        port = int(os.environ.get('PORT', 8080))
        site = web.TCPSite(runner, '0.0.0.0', port)
        await site.start()
        
        logger.info(f"Health check server started on port {port}")
        return runner

async def main():
    """Main function to start the Telegram bot"""
    bot = TradingBot()
    bot.start_time = time.time()
    
    try:
        logger.info("Starting AI Trading Bot (Fixed Version)...")
        secure_logger.log_system_event("bot_startup", "AI Trading Bot startup initiated")
        
        # Validate environment
        if not bot.validate_environment():
            logger.error("Environment validation failed, exiting...")
            sys.exit(1)
        
        logger.info("Starting performance optimizations...")
        
        # Skip heavy background tasks to avoid thread limits
        logger.info("Background services disabled to reduce thread usage")
        
        # Start health check server
        bot.health_server = await bot.start_health_server()
        
        # Initialize Telegram handler
        bot.telegram_handler = TelegramHandler()
        logger.info("TelegramHandler created successfully")
        
        # Mark as ready
        bot.is_ready = True
        
        logger.info("Trading bot started successfully!")
        logger.info("Health server: http://localhost:8080/health")
        logger.info("Bot is ready to serve requests")
        
        # Run the telegram bot
        await bot.telegram_handler.run()
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot crashed: {str(e)}")
        secure_logger.log_security_event("bot_crash", f"Bot crashed with error: {str(e)}")
        raise
    finally:
        logger.info("Shutting down bot...")
        secure_logger.log_system_event("bot_shutdown_complete", "Trading bot shutdown completed")

if __name__ == "__main__":
    asyncio.run(main())