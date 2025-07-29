#!/usr/bin/env python3
"""
Enhanced startup script with dependency management
"""
import sys
import os
import asyncio
import logging

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure minimal logging to reduce overhead
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('TradingBot')

async def enhanced_main():
    """Enhanced main function with dependency checks"""
    try:
        logger.info("Starting AI Trading Bot (Enhanced Mode)...")
        
        # Check for numpy first
        try:
            import numpy as np
            logger.info(f"NumPy {np.__version__} available")
        except ImportError:
            logger.error("NumPy not available - using minimal mode")
            # Fall back to minimal bot
            from minimal_start import MinimalTradingBot
            bot = MinimalTradingBot()
            await bot.run()
            return
        
        # Try to import the full telegram handler
        try:
            from telegram_handler import TelegramHandler
            logger.info("Full TelegramHandler imported successfully")
            
            # Create and run the enhanced bot
            handler = TelegramHandler()
            await handler.run()
            
        except ImportError as e:
            logger.warning(f"Full handler not available: {e}")
            logger.info("Falling back to minimal mode...")
            
            # Fall back to minimal bot
            from minimal_start import MinimalTradingBot
            bot = MinimalTradingBot()
            await bot.run()
            
    except Exception as e:
        logger.error(f"Bot crashed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(enhanced_main())
