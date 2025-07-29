"""
Minimal main entry point for deployment - no heavy dependencies
"""
import sys
import os
import signal
import asyncio
from aiohttp import web
import time
from typing import Optional

# Print environment info
print("=== Minimal Bot Startup ===")
print(f"Python: {sys.version}")
print(f"Directory: {os.getcwd()}")

# Ensure the current directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Core imports only
from logger import logger
from config import Config
from telegram_handler import TelegramHandler

class MinimalTradingBot:
    """Minimal bot class for deployment"""
    
    def __init__(self):
        self.telegram_handler: Optional[TelegramHandler] = None
        self.health_server: Optional[web.AppRunner] = None
        self.is_ready: bool = False
        self.start_time: Optional[float] = None
        logger.info("Minimal Trading Bot initializing...")
    
    def setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info("Received shutdown signal, stopping bot...")
            if self.health_server:
                asyncio.create_task(self.health_server.cleanup())
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def health_check(self, request):
        """Health check endpoint"""
        return web.Response(text="OK", status=200)
    
    async def readiness_check(self, request):
        """Readiness check endpoint"""
        if self.is_ready and self.telegram_handler:
            return web.Response(text="Ready", status=200)
        else:
            return web.Response(text="Not Ready", status=503)
    
    async def start_health_server(self):
        """Start the health check HTTP server"""
        app = web.Application()
        
        async def root_handler(request):
            return web.Response(text="TradeAI Companion Bot is running!", status=200)
        
        app.router.add_get('/health', self.health_check)
        app.router.add_get('/ready', self.readiness_check)
        app.router.add_get('/', root_handler)
        
        runner = web.AppRunner(app)
        await runner.setup()
        
        # Use PORT environment variable for deployment
        port = int(os.environ.get('PORT', 5000))
        site = web.TCPSite(runner, '0.0.0.0', port)
        await site.start()
        
        logger.info(f"Health server started on port {port}")
        return runner

async def main():
    """Main function to start the minimal bot"""
    bot = MinimalTradingBot()
    bot.start_time = time.time()
    
    try:
        logger.info("Starting Minimal AI Trading Bot...")
        
        # Setup signal handlers
        bot.setup_signal_handlers()
        
        # Start health check server
        try:
            bot.health_server = await bot.start_health_server()
            logger.info("Health server started successfully")
        except Exception as e:
            logger.warning(f"Health server failed: {e}, continuing without it")
        
        # Initialize Telegram handler
        try:
            bot.telegram_handler = TelegramHandler()
            logger.info("TelegramHandler created successfully")
        except Exception as e:
            logger.error(f"Failed to create TelegramHandler: {e}")
            raise
        
        # Mark as ready
        bot.is_ready = True
        
        logger.info("🤖 Minimal trading bot started successfully!")
        port = int(os.environ.get('PORT', 5000))
        logger.info(f"Health server: http://localhost:{port}/health")
        logger.info("Bot is ready to serve requests")
        
        # Start the bot
        await bot.telegram_handler.run()
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot crashed: {e}")
        sys.exit(1)
    finally:
        if bot.health_server:
            await bot.health_server.cleanup()
            logger.info("Health server stopped")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            logger.info("Already in event loop, creating task...")
            asyncio.create_task(main())
        else:
            raise