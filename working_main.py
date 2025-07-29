#!/usr/bin/env python3
"""
Working version of main.py that properly starts the bot
"""
import sys
import os
import asyncio
import logging

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import the minimal bot that we know works
from minimal_start import MinimalTradingBot

async def main():
    """Run the minimal bot that we know works"""
    try:
        print("Starting Working Bot...")
        bot = MinimalTradingBot()
        await bot.run()
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())