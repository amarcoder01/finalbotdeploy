"""
Main entry point for the Telegram AI Trading Bot
Initializes and starts the bot with proper error handling
"""
import sys
import os
import signal
import asyncio
from aiohttp import web
import threading
import time
from typing import Optional

# Print the sys.path for debugging environment issues
print("--- sys.path ---")
print(sys.path)
print("----------------")

# Ensure the current directory is in sys.path for module imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logger import logger
from config import Config
from telegram_handler import TelegramHandler
from monitoring import metrics, start_metrics_server, setup_periodic_cleanup
from performance_cache import (
    performance_cache, response_cache, connection_pool, preloader,
    get_cache_stats, clear_all_caches
)
from security_config import SecurityConfig, SecurityError
from secure_logger import secure_logger
from rate_limiter import RateLimiter
from input_validator import InputValidator

class TradingBot:
    """Main bot class that orchestrates all components"""
    
    def __init__(self):
        """Initialize the trading bot"""
        self.telegram_handler: Optional[TelegramHandler] = None
        self.health_server: Optional[web.AppRunner] = None
        self.is_ready: bool = False
        self.config: Optional[Config] = None
        self.security_config: Optional[SecurityConfig] = None
        self.rate_limiter: Optional[RateLimiter] = None
        self.input_validator: Optional[InputValidator] = None
        self.start_time: Optional[float] = None
        logger.info("Trading Bot initializing...")
        secure_logger.log_system_event("bot_initialization", "Trading bot initialization started")
    
    def validate_environment(self) -> bool:
        """
        Validate environment and configuration
        
        Returns:
            bool: True if environment is valid, False otherwise
        """
        try:
            # Initialize configuration with security
            self.config = Config.validate_required_configs()
            self.security_config = self.config.security
            
            # Initialize security components
            self.rate_limiter = RateLimiter()
            self.input_validator = InputValidator()
            
            logger.info("Environment validation successful")
            secure_logger.log_system_event("environment_validation", "Environment and security validation completed")
            return True
        except (ValueError, SecurityError) as e:
            logger.error(f"Environment validation failed: {str(e)}")
            secure_logger.log_security_event("environment_validation_failed", f"Validation failed: {str(e)}")
            return False
    
    def setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info("Received shutdown signal, stopping bot...")
            if self.health_server:
                self.health_server.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def health_check(self, request):
        """Health check endpoint for load balancer"""
        return web.Response(text="OK", status=200)
    
    async def readiness_check(self, request):
        """Readiness check endpoint for load balancer"""
        if self.is_ready and self.telegram_handler:
            return web.Response(text="Ready", status=200)
        else:
            return web.Response(text="Not Ready", status=503)
    
    async def metrics_endpoint(self, request):
        """Metrics endpoint for monitoring"""
        try:
            # Get bot metrics
            bot_metrics = {
                'uptime_seconds': time.time() - self.start_time,
                'ready': self.is_ready,
                'telegram_handler_status': 'active' if self.telegram_handler else 'inactive'
            }
            
            # Get Prometheus metrics
            prometheus_metrics = metrics.get_metrics_summary()
            
            # Get cache statistics
            cache_stats = get_cache_stats()
            
            # Combine metrics
            combined_metrics = {
                **bot_metrics,
                **prometheus_metrics,
                'cache_stats': cache_stats
            }
            
            return web.json_response(combined_metrics)
        except Exception as e:
            logger.error(f"Error in metrics endpoint: {e}")
            return web.json_response({'error': 'Metrics unavailable'}, status=500)
    
    async def start_health_server(self):
        """Start the health check HTTP server"""
        app = web.Application()
        
        async def root_handler(request):
            return web.Response(text="TradeAI Companion Bot is running!", status=200)
        
        app.router.add_get('/health', self.health_check)
        app.router.add_get('/ready', self.readiness_check)
        app.router.add_get('/metrics', self.metrics_endpoint)
        app.router.add_get('/', root_handler)
        
        runner = web.AppRunner(app)
        await runner.setup()
        
        # Use PORT environment variable for deployment, default to 5000
        port = int(os.environ.get('PORT', 5000))
        site = web.TCPSite(runner, '0.0.0.0', port)
        await site.start()
        
        logger.info(f"Health check server started on port {port}")
        return runner
    
async def main():
    """Main function to start the Telegram bot (for local development only)"""
    # Check if we're running on Render (has PORT env var)
    if os.environ.get('PORT'):
        logger.info("Detected Render deployment, skipping main() - using aiohttp.web instead")
        return
    
    bot = TradingBot()
    bot.start_time = time.time()
    
    try:
        logger.info("Starting AI Trading Bot...")
        secure_logger.log_system_event("bot_startup", "AI Trading Bot startup initiated")
        
        # Validate environment and initialize security first
        if not bot.validate_environment():
            logger.error("Environment validation failed, exiting...")
            secure_logger.log_security_event("startup_failed", "Bot startup failed due to environment validation")
            sys.exit(1)
        
        logger.info("🚀 Starting performance optimizations...")
        
        # Skip preloading to avoid hang
        logger.info("Preloading disabled to avoid startup hang")
        
        # Skip metrics server to avoid thread limits
        logger.info("Metrics server disabled to reduce thread usage")
        
        # Start health check server
        try:
            bot.health_server = await bot.start_health_server()
        except Exception as e:
            logger.warning(f"Health server failed: {e}, continuing without it")
        
        # Initialize Telegram handler with security components
        bot.telegram_handler = TelegramHandler()
        logger.info("TelegramHandler created successfully")
        secure_logger.log_system_event("telegram_handler_initialized", "Telegram handler created with security middleware")
        
        # Preloading skipped
        
        # Mark as ready
        bot.is_ready = True
        metrics.set_ready_status(True)
        
        # Log cache statistics
        cache_stats = get_cache_stats()
        logger.info(f"📊 Cache initialized: {cache_stats['performance_cache']['entries']} entries")
        
        logger.info("🤖 Trading bot started successfully with performance optimizations and security!")
        logger.info(f"Health server: http://localhost:{port}/health" if 'port' in locals() else "Health server: http://localhost:5000/health")
        logger.info("Metrics server: http://localhost:9090")
        logger.info(f"Connection pool: {connection_pool.max_connections} max connections")
        logger.info("🔒 Security features: Rate limiting, Input validation, Secure logging, Session management")
        logger.info("Bot is ready to serve requests")
        
        secure_logger.log_system_event("bot_startup_complete", "Trading bot startup completed successfully with all security features enabled")
        
        # Start the bot
        await bot.telegram_handler.run()
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
        secure_logger.log_system_event("bot_shutdown_user", "Bot stopped by user interrupt")
    except Exception as e:
        logger.error(f"Bot crashed: {e}")
        secure_logger.log_security_event("bot_crash", f"Bot crashed with error: {str(e)}", severity="critical")
        sys.exit(1)
    finally:
        if bot.health_server:
            await bot.health_server.cleanup()
            logger.info("Health server stopped")
            secure_logger.log_system_event("health_server_stopped", "Health check server shutdown completed")
        
        # Log final shutdown
        secure_logger.log_system_event("bot_shutdown_complete", "Trading bot shutdown completed")

# Create a global app instance for Render deployment
_app_instance = None
bot_instance = None

def create_app():
    """Create and return the web application for deployment"""
    global _app_instance, bot_instance
    if _app_instance is None:
        from aiohttp import web
        import logging
        
        # Configure logging for production
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create application with proper middleware for production
        _app_instance = web.Application(
            middlewares=[],
            client_max_size=1024**2  # 1MB max request size
        )
        
        # Initialize bot instance
        bot_instance = TradingBot()
        bot_instance.start_time = time.time()
        
        # Add routes
        async def root_handler(request):
            return web.Response(
                text="TradeAI Companion Bot is running! Health: /health, Ready: /ready, Metrics: /metrics", 
                status=200,
                headers={'Content-Type': 'text/plain'}
            )
        
        async def favicon_handler(request):
            return web.Response(status=404)
        
        _app_instance.router.add_get('/health', bot_instance.health_check)
        _app_instance.router.add_get('/ready', bot_instance.readiness_check)
        _app_instance.router.add_get('/metrics', bot_instance.metrics_endpoint)
        _app_instance.router.add_get('/', root_handler)
        _app_instance.router.add_get('/favicon.ico', favicon_handler)
        
        # Initialize bot in background
        async def init_bot():
            try:
                logger.info("Initializing bot for web deployment...")
                secure_logger.log_system_event("bot_startup", "AI Trading Bot startup initiated for web deployment")
                
                if not bot_instance.validate_environment():
                    logger.error("Environment validation failed")
                    secure_logger.log_security_event("startup_failed", "Bot startup failed due to environment validation")
                    return
                
                logger.info("🚀 Starting performance optimizations...")
                
                # Start background services with error handling
                try:
                    start_metrics_server(port=9090)
                    setup_periodic_cleanup()
                    logger.info("Background services started")
                except Exception as e:
                    logger.warning(f"Background services failed to start: {e}")
                
                # Note: Health server is NOT started here because aiohttp.web handles the web server
                # The health endpoints are already registered in the main app routes
                logger.info("Health endpoints available via main web server (no separate health server needed)")
                
                # Initialize Telegram handler with security components
                bot_instance.telegram_handler = TelegramHandler()
                logger.info("TelegramHandler created successfully")
                secure_logger.log_system_event("telegram_handler_initialized", "Telegram handler created with security middleware")
                
                # Mark as ready
                bot_instance.is_ready = True
                metrics.set_ready_status(True)
                
                # Log cache statistics
                cache_stats = get_cache_stats()
                logger.info(f"📊 Cache initialized: {cache_stats['performance_cache']['entries']} entries")
                
                logger.info("🤖 Trading bot started successfully with performance optimizations and security!")
                logger.info("🔒 Security features: Rate limiting, Input validation, Secure logging, Session management")
                logger.info("Bot is ready to serve requests")
                
                secure_logger.log_system_event("bot_startup_complete", "Trading bot startup completed successfully with all security features enabled")
                
                # Start the telegram bot in background
                asyncio.create_task(bot_instance.telegram_handler.run())
                
            except Exception as e:
                logger.error(f"Bot initialization failed: {e}")
                secure_logger.log_security_event("bot_crash", f"Bot initialization failed: {str(e)}", severity="critical")
                # Don't fail the web app if bot fails
                bot_instance.is_ready = False
        
        # Schedule bot initialization when event loop is available
        async def startup_handler(app):
            logger.info("Web application startup initiated")
            # Use create_task to avoid blocking the startup
            asyncio.create_task(init_bot_background())
        
        # Non-blocking bot initialization
        async def init_bot_background():
            """Initialize bot in background without blocking web server startup"""
            try:
                await asyncio.sleep(1)  # Small delay to ensure web server is ready
                await init_bot()
            except Exception as e:
                logger.error(f"Background bot initialization failed: {e}")
        
        # Graceful shutdown
        async def cleanup_handler(app):
            logger.info("Shutting down application...")
            secure_logger.log_system_event("bot_shutdown", "Trading bot shutdown initiated")
            
            if bot_instance and bot_instance.telegram_handler:
                try:
                    await bot_instance.telegram_handler.stop()
                    logger.info("Telegram handler stopped")
                except Exception as e:
                    logger.error(f"Error stopping telegram handler: {e}")
            
            # Log final shutdown
            secure_logger.log_system_event("bot_shutdown_complete", "Trading bot shutdown completed")
        
        _app_instance.on_startup.append(startup_handler)
        _app_instance.on_cleanup.append(cleanup_handler)
        
        # Log application creation for debugging
        logger.info(f"Web application created successfully. Type: {type(_app_instance)}")
        logger.info(f"Application routes: {len(_app_instance.router.routes())} routes registered")
    
    return _app_instance

if __name__ == "__main__":
    try:
        # Use get_event_loop() to avoid conflicts with existing event loops
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running, create a task instead
            task = loop.create_task(main())
            loop.run_until_complete(task)
        else:
            # If no loop is running, use asyncio.run
            asyncio.run(main())
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            # Handle the case where we're already in an event loop
            logger.info("Already in event loop, creating task...")
            asyncio.create_task(main())
        else:
            raise

# For Render deployment - aiohttp.web expects a factory function
def app(argv=None):
    """Factory function for aiohttp.web deployment"""
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("App factory called by aiohttp.web")
    logger.info(f"Arguments: {argv}")
    
    app_instance = create_app()
    logger.info(f"Created app instance: {type(app_instance)}")
    logger.info(f"App has {len(list(app_instance.router.routes()))} routes")
    
    return app_instance

# Alternative entry point for direct web server deployment
async def init_app():
    """Initialize and return the web application"""
    return create_app()

# For gunicorn/uvicorn deployment
def get_app():
    """Get the application instance for WSGI/ASGI servers"""
    return create_app()

# Direct app instance for immediate access
app_instance = create_app()
