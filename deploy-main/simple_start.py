#!/usr/bin/env python3
"""
Simple startup script with minimal background services
"""
import sys
import os
import asyncio
import logging

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from telegram_handler import TelegramHandler
from logger import logger

async def simple_main():
    """Simplified main function with minimal services"""
    try:
        logger.info("Starting AI Trading Bot (Simple Mode)...")
        
        # Create and start bot with minimal configuration
        handler = TelegramHandler()
        
        # Use the existing run method
        await handler.run()
            
    except Exception as e:
        logger.error(f"Bot crashed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(simple_main())
