#!/usr/bin/env python3
"""
Database migration script to fix user_id column type from INTEGER to BIGINT
This fixes the issue where Telegram chat IDs exceed the 32-bit integer range
"""

import asyncio
from sqlalchemy import text
from db import engine
from logger import BotLogger

logger = BotLogger(__name__)

async def migrate_user_id_to_bigint():
    """
    Migrate user_id columns from INTEGER to BIGINT in user_memories and memory_insights tables
    """
    try:
        logger.info("Starting user_id BIGINT migration...")
        
        async with engine.begin() as connection:
            # Check if tables exist
            tables_check = await connection.execute(text("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('user_memories', 'memory_insights')
            """))
            existing_tables = [row[0] for row in tables_check.fetchall()]
            
            # Migrate user_memories table
            if 'user_memories' in existing_tables:
                logger.info("Migrating user_memories.user_id to BIGINT...")
                
                # Check current column type
                column_check = await connection.execute(text("""
                    SELECT data_type FROM information_schema.columns 
                    WHERE table_name = 'user_memories' AND column_name = 'user_id'
                """))
                current_type = column_check.fetchone()
                
                if current_type and current_type[0] == 'integer':
                    # Alter the column type
                    await connection.execute(text("""
                        ALTER TABLE user_memories 
                        ALTER COLUMN user_id TYPE BIGINT
                    """))
                    logger.info("✓ user_memories.user_id migrated to BIGINT")
                else:
                    logger.info("user_memories.user_id already BIGINT or table doesn't exist")
            else:
                logger.info("user_memories table doesn't exist, will be created with BIGINT")
            
            # Migrate memory_insights table
            if 'memory_insights' in existing_tables:
                logger.info("Migrating memory_insights.user_id to BIGINT...")
                
                # Check current column type
                column_check = await connection.execute(text("""
                    SELECT data_type FROM information_schema.columns 
                    WHERE table_name = 'memory_insights' AND column_name = 'user_id'
                """))
                current_type = column_check.fetchone()
                
                if current_type and current_type[0] == 'integer':
                    # Alter the column type
                    await connection.execute(text("""
                        ALTER TABLE memory_insights 
                        ALTER COLUMN user_id TYPE BIGINT
                    """))
                    logger.info("✓ memory_insights.user_id migrated to BIGINT")
                else:
                    logger.info("memory_insights.user_id already BIGINT or table doesn't exist")
            else:
                logger.info("memory_insights table doesn't exist, will be created with BIGINT")
            
            logger.info("✓ User ID BIGINT migration completed successfully")
            
    except Exception as e:
        logger.error(f"Error during user_id BIGINT migration: {e}")
        raise

async def main():
    """Run the migration"""
    try:
        await migrate_user_id_to_bigint()
        print("Migration completed successfully!")
    except Exception as e:
        print(f"Migration failed: {e}")
        return 1
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))