import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from db import AsyncSessionLocal
from sqlalchemy import text

async def test_alert_functionality():
    """Simple test to verify alert functionality after migration"""
    async with AsyncSessionLocal() as session:
        try:
            print("\n=== Testing Alert Functionality ===")
            
            # Test 1: Check if users table has access_level column
            print("\n1. Checking users table structure...")
            result = await session.execute(
                text("SELECT column_name FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'access_level'")
            )
            if result.fetchone():
                print("✓ users.access_level column exists")
            else:
                print("✗ users.access_level column missing")
            
            # Test 2: Check if alerts table has created_from_ip column
            print("\n2. Checking alerts table structure...")
            result = await session.execute(
                text("SELECT column_name FROM information_schema.columns WHERE table_name = 'alerts' AND column_name = 'created_from_ip'")
            )
            if result.fetchone():
                print("✓ alerts.created_from_ip column exists")
            else:
                print("✗ alerts.created_from_ip column missing")
            
            # Test 3: Try to insert a test user with access_level
            print("\n3. Testing user insertion with access_level...")
            try:
                await session.execute(
                    text("INSERT INTO users (telegram_id, access_level) VALUES (:telegram_id, :access_level) ON CONFLICT (telegram_id) DO UPDATE SET access_level = EXCLUDED.access_level"),
                    {"telegram_id": "999999999", "access_level": "user"}
                )
                await session.commit()
                print("✓ User insertion with access_level successful")
            except Exception as e:
                print(f"✗ User insertion failed: {e}")
                await session.rollback()
            
            # Get the user ID for alert insertion
            print("\n4. Getting user ID for alert test...")
            try:
                result = await session.execute(
                    text("SELECT id FROM users WHERE telegram_id = :telegram_id"),
                    {"telegram_id": "999999999"}
                )
                user_row = result.fetchone()
                if user_row:
                    user_id = user_row[0]
                    print(f"✓ Found user ID: {user_id}")
                    
                    # Test 5: Try to insert a test alert with created_from_ip
                    print("\n5. Testing alert insertion with created_from_ip...")
                    try:
                        await session.execute(
                            text("INSERT INTO alerts (user_id, symbol, condition, created_from_ip) VALUES (:user_id, :symbol, :condition, :ip)"),
                            {"user_id": user_id, "symbol": "TEST", "condition": "above", "ip": "127.0.0.1"}
                        )
                        await session.commit()
                        print("✓ Alert insertion with created_from_ip successful")
                    except Exception as e:
                        print(f"✗ Alert insertion failed: {e}")
                        await session.rollback()
                else:
                    print("✗ Could not find test user")
            except Exception as e:
                print(f"✗ Failed to get user ID: {e}")
            
            # Test 6: Clean up test data
            print("\n6. Cleaning up test data...")
            try:
                await session.execute(text("DELETE FROM alerts WHERE symbol = 'TEST'"))
                await session.execute(text("DELETE FROM users WHERE telegram_id = '999999999'"))
                await session.commit()
                print("✓ Test data cleaned up")
            except Exception as e:
                print(f"✗ Cleanup failed: {e}")
                await session.rollback()
            
            print("\n=== Test Complete ===")
            
        except Exception as e:
            print(f"Test failed with error: {e}")

if __name__ == "__main__":
    asyncio.run(test_alert_functionality())