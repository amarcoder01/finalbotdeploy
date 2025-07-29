#!/usr/bin/env python3
"""
Debug script to test alert removal functionality
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from db import AsyncSessionLocal
from sqlalchemy import text
from alert_service import AlertService
from market_data_service import MarketDataService

async def debug_alert_removal():
    """Debug alert removal issue"""
    try:
        print("\n=== Debugging Alert Removal ===\n")
        
        # Initialize services
        market_service = MarketDataService()
        alert_service = AlertService(market_service)
        
        # Test user ID from the image (assuming it's the user trying to remove alert 14)
        test_user_telegram_id = 123456789  # Replace with actual user ID if known
        alert_id_to_remove = 14
        
        print(f"Testing removal of alert ID {alert_id_to_remove}...\n")
        
        # First, let's check what alerts exist in the database
        async with AsyncSessionLocal() as session:
            print("1. Checking all alerts in database:")
            result = await session.execute(text("SELECT id, user_id, symbol, condition, is_active FROM alerts ORDER BY id"))
            alerts = result.fetchall()
            
            if alerts:
                for alert in alerts:
                    print(f"   Alert ID: {alert[0]}, User ID: {alert[1]}, Symbol: {alert[2]}, Condition: {alert[3]}, Active: {alert[4]}")
            else:
                print("   No alerts found in database")
            
            print("\n2. Checking users table:")
            result = await session.execute(text("SELECT id, telegram_id FROM users ORDER BY id"))
            users = result.fetchall()
            
            if users:
                for user in users:
                    print(f"   User ID: {user[0]}, Telegram ID: {user[1]}")
            else:
                print("   No users found in database")
            
            # Check if alert 14 exists
            print(f"\n3. Checking if alert {alert_id_to_remove} exists:")
            result = await session.execute(text(f"SELECT id, user_id, symbol, condition FROM alerts WHERE id = {alert_id_to_remove}"))
            alert_14 = result.fetchone()
            
            if alert_14:
                print(f"   ✓ Alert {alert_id_to_remove} found: User ID {alert_14[1]}, Symbol: {alert_14[2]}, Condition: {alert_14[3]}")
                
                # Get the user's telegram_id
                result = await session.execute(text(f"SELECT telegram_id FROM users WHERE id = {alert_14[1]}"))
                user_telegram = result.fetchone()
                
                if user_telegram:
                    actual_telegram_id = int(user_telegram[0])
                    print(f"   ✓ Alert belongs to Telegram user: {actual_telegram_id}")
                    
                    # Test removal with the correct telegram ID
                    print(f"\n4. Testing alert removal with correct Telegram ID {actual_telegram_id}:")
                    removal_result = await alert_service.remove_alert(actual_telegram_id, alert_id_to_remove)
                    print(f"   Result: {removal_result}")
                    
                    if removal_result['success']:
                        print("   ✓ Alert removal successful!")
                        
                        # Verify it's actually gone
                        print("\n5. Verifying alert is removed:")
                        result = await session.execute(text(f"SELECT id FROM alerts WHERE id = {alert_id_to_remove}"))
                        still_exists = result.fetchone()
                        
                        if still_exists:
                            print("   ✗ Alert still exists in database!")
                        else:
                            print("   ✓ Alert successfully removed from database")
                    else:
                        print(f"   ✗ Alert removal failed: {removal_result.get('error', 'Unknown error')}")
                else:
                    print(f"   ✗ Could not find user with ID {alert_14[1]}")
            else:
                print(f"   ✗ Alert {alert_id_to_remove} not found in database")
                
                # Let's also test with a different approach - direct SQL
                print(f"\n6. Testing direct SQL deletion of alert {alert_id_to_remove}:")
                try:
                    result = await session.execute(text(f"DELETE FROM alerts WHERE id = {alert_id_to_remove}"))
                    await session.commit()
                    print(f"   ✓ Direct SQL deletion successful, rows affected: {result.rowcount}")
                except Exception as e:
                    print(f"   ✗ Direct SQL deletion failed: {e}")
                    await session.rollback()
        
        print("\n=== Debug Complete ===\n")
        
    except Exception as e:
        print(f"Debug script error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_alert_removal())