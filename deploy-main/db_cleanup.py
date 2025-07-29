import asyncio
import asyncpg
import os
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

# Convert SQLAlchemy URL to asyncpg URL if needed
if DATABASE_URL and DATABASE_URL.startswith("postgresql+asyncpg"):
    DATABASE_URL = DATABASE_URL.replace("postgresql+asyncpg", "postgresql")

async def drop_tables():
    conn = await asyncpg.connect(DATABASE_URL, ssl=True)
    for table in ["predictions", "stocks", "alerts", "trades", "users"]:
        try:
            await conn.execute(f'DROP TABLE IF EXISTS {table} CASCADE;')
            print(f"✅ Dropped table: {table}")
        except Exception as e:
            print(f"❌ Error dropping {table}: {e}")
    await conn.close()

if __name__ == "__main__":
    asyncio.run(drop_tables()) 