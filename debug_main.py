#!/usr/bin/env python3
"""
Debug version of main.py to identify where it's getting stuck
"""
import sys
import os
import asyncio
import time
import logging

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('DebugBot')

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock missing dependencies
class FakeModule:
    def __getattr__(self, name):
        return FakeModule()
    def __call__(self, *args, **kwargs):
        return FakeModule()
    def __getitem__(self, key):
        return FakeModule()

# Mock numpy before any imports
try:
    import numpy
except ImportError:
    logger.warning("NumPy not available, creating mock")
    class FakeNumPy:
        __version__ = '1.24.0'
        def __getattr__(self, name):
            return FakeModule()
    fake_numpy = FakeNumPy()
    sys.modules['numpy'] = fake_numpy
    sys.modules['numpy.linalg'] = fake_numpy
    sys.modules['numpy.core'] = fake_numpy
    sys.modules['numpy.core.multiarray'] = fake_numpy
    sys.modules['numpy.core._multiarray_umath'] = fake_numpy

# Mock pandas before import
try:
    import pandas
except ImportError:
    logger.warning("Pandas not available, mocking it")
    # Skip pandas entirely - it's not critical
    pass

async def main():
    """Debug main function"""
    try:
        logger.info("=== Starting Debug Bot ===")
        
        # Check environment
        token = os.getenv('TELEGRAM_API_TOKEN')
        if not token:
            logger.error("TELEGRAM_API_TOKEN not set!")
            return
        logger.info("✓ Telegram token found")
        
        # Import telegram handler
        logger.info("Importing TelegramHandler...")
        from telegram_handler import TelegramHandler
        logger.info("✓ TelegramHandler imported")
        
        # Create handler
        logger.info("Creating TelegramHandler instance...")
        handler = TelegramHandler()
        logger.info("✓ TelegramHandler created")
        
        # Run bot
        logger.info("Starting bot.run()...")
        await handler.run()
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)

if __name__ == "__main__":
    logger.info("Starting asyncio event loop...")
    asyncio.run(main())