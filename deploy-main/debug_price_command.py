#!/usr/bin/env python3

import asyncio
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the project directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from telegram_handler import TelegramHandler
from market_data_service import MarketDataService

async def debug_price_command():
    """Debug the price command flow for AAPL"""
    print("=== Debugging Price Command for AAPL ===")
    
    try:
        # Initialize services
        print("\n1. Initializing services...")
        market_service = MarketDataService()
        
        # Create a minimal telegram handler instance for testing validation
        telegram_handler = TelegramHandler.__new__(TelegramHandler)
        telegram_handler.market_service = market_service
        telegram_handler.us_stocks_cache = None  # Will trigger loading
        
        print("✅ Services initialized successfully")
        
        # Test symbol validation
        print("\n2. Testing symbol validation...")
        symbol = "AAPL"
        is_valid = telegram_handler._is_valid_us_stock(symbol)
        print(f"Is '{symbol}' valid? {is_valid}")
        
        # Test symbol normalization
        print("\n3. Testing symbol normalization...")
        normalized = telegram_handler._normalize_us_stock_symbol(symbol)
        print(f"Normalized '{symbol}' -> '{normalized}'")
        
        # Test market data service directly
        print("\n4. Testing market data service...")
        try:
            price_data = await market_service.get_stock_price(normalized, user_id="test_user")
            if price_data:
                print(f"✅ Price data retrieved: {price_data}")
            else:
                print("❌ No price data returned")
        except Exception as e:
            print(f"❌ Market data service error: {e}")
        
        # Test the complete price command logic
        print("\n5. Testing complete price command logic...")
        
        # Simulate the price command validation and processing
        if not telegram_handler._is_valid_us_stock(symbol):
            print(f"❌ Symbol validation failed for {symbol}")
            return
        
        normalized_symbol = telegram_handler._normalize_us_stock_symbol(symbol)
        print(f"Symbol after normalization: {normalized_symbol}")
        
        try:
            # This is the exact call made in price_command
            price_data = await market_service.get_stock_price(normalized_symbol, user_id="test_user")
            
            if not price_data:
                print("❌ No price data returned from market service")
                return
            
            print(f"✅ Price command would succeed with data: {price_data}")
            
        except Exception as e:
            print(f"❌ Error in price command logic: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"❌ Error during debugging: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_price_command())