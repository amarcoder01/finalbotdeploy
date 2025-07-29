#!/usr/bin/env python3
"""
Security Migration Script
Adds security-related fields to existing database tables
"""

import sys
import os
import asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from db import AsyncSessionLocal, DATABASE_URL
from models import Base
from logger import logger
from secure_logger import secure_logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create async engine for migrations
async_engine = create_async_engine(DATABASE_URL)

async def run_security_migration():
    """
    Run security migration to add new fields to existing tables
    """
    try:
        logger.info("Starting security migration...")
        secure_logger.log_system_event("migration_start", "Security database migration initiated")
        
        async with async_engine.begin() as connection:
            try:
                # Add security fields to users table
                logger.info("Adding security fields to users table...")
                
                # Check if columns already exist before adding them
                user_columns = [
                    "ALTER TABLE users ADD COLUMN access_level VARCHAR DEFAULT 'user'",
                    "ALTER TABLE users ADD COLUMN is_active BOOLEAN DEFAULT true",
                    "ALTER TABLE users ADD COLUMN failed_login_attempts INTEGER DEFAULT 0",
                    "ALTER TABLE users ADD COLUMN last_login_attempt TIMESTAMP",
                    "ALTER TABLE users ADD COLUMN last_successful_login TIMESTAMP",
                    "ALTER TABLE users ADD COLUMN current_session_id VARCHAR",
                    "ALTER TABLE users ADD COLUMN session_expires_at TIMESTAMP",
                    "ALTER TABLE users ADD COLUMN security_events_count INTEGER DEFAULT 0",
                    "ALTER TABLE users ADD COLUMN last_security_event TIMESTAMP",
                    "ALTER TABLE users ADD COLUMN encrypted_data TEXT"
                ]
                
                for sql in user_columns:
                    try:
                        await connection.execute(text(sql))
                        logger.info(f"Added column: {sql.split('ADD COLUMN')[1].split()[0]}")
                    except Exception as e:
                        if "already exists" in str(e).lower() or "duplicate column" in str(e).lower():
                            logger.info(f"Column already exists, skipping: {sql.split('ADD COLUMN')[1].split()[0]}")
                        else:
                            raise e
                
                # Add security fields to alerts table
                logger.info("Adding security fields to alerts table...")
                
                alert_columns = [
                    "ALTER TABLE alerts ADD COLUMN created_from_ip VARCHAR",
                    "ALTER TABLE alerts ADD COLUMN last_triggered TIMESTAMP",
                    "ALTER TABLE alerts ADD COLUMN trigger_count INTEGER DEFAULT 0"
                ]
                
                for sql in alert_columns:
                    try:
                        await connection.execute(text(sql))
                        logger.info(f"Added column: {sql.split('ADD COLUMN')[1].split()[0]}")
                    except Exception as e:
                        if "already exists" in str(e).lower() or "duplicate column" in str(e).lower():
                            logger.info(f"Column already exists, skipping: {sql.split('ADD COLUMN')[1].split()[0]}")
                        else:
                            raise e
                
                # Add security fields to trades table
                logger.info("Adding security fields to trades table...")
                
                trade_columns = [
                    "ALTER TABLE trades ADD COLUMN created_from_ip VARCHAR",
                    "ALTER TABLE trades ADD COLUMN validation_status VARCHAR DEFAULT 'pending'",
                    "ALTER TABLE trades ADD COLUMN risk_score FLOAT DEFAULT 0.0"
                ]
                
                for sql in trade_columns:
                    try:
                        await connection.execute(text(sql))
                        logger.info(f"Added column: {sql.split('ADD COLUMN')[1].split()[0]}")
                    except Exception as e:
                        if "already exists" in str(e).lower() or "duplicate column" in str(e).lower():
                            logger.info(f"Column already exists, skipping: {sql.split('ADD COLUMN')[1].split()[0]}")
                        else:
                            raise e
                
                # Create security_logs table
                logger.info("Creating security_logs table...")
                
                create_security_logs = """
                CREATE TABLE IF NOT EXISTS security_logs (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER REFERENCES users(id),
                    event_type VARCHAR NOT NULL,
                    event_details TEXT,
                    ip_address VARCHAR,
                    user_agent VARCHAR,
                    severity VARCHAR DEFAULT 'info',
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    session_id VARCHAR,
                    endpoint VARCHAR,
                    success BOOLEAN DEFAULT true,
                    error_message TEXT
                )
                """
                
                await connection.execute(text(create_security_logs))
                logger.info("Security logs table created successfully")
                
                # Create indexes for performance
                logger.info("Creating security indexes...")
                
                indexes = [
                    "CREATE INDEX IF NOT EXISTS idx_users_telegram_id ON users(telegram_id)",
                    "CREATE INDEX IF NOT EXISTS idx_users_access_level ON users(access_level)",
                    "CREATE INDEX IF NOT EXISTS idx_users_session_id ON users(current_session_id)",
                    "CREATE INDEX IF NOT EXISTS idx_security_logs_user_id ON security_logs(user_id)",
                    "CREATE INDEX IF NOT EXISTS idx_security_logs_event_type ON security_logs(event_type)",
                    "CREATE INDEX IF NOT EXISTS idx_security_logs_timestamp ON security_logs(timestamp)",
                    "CREATE INDEX IF NOT EXISTS idx_security_logs_severity ON security_logs(severity)"
                ]
                
                for index_sql in indexes:
                    try:
                        await connection.execute(text(index_sql))
                        logger.info(f"Created index: {index_sql.split('idx_')[1].split()[0] if 'idx_' in index_sql else 'index'}")
                    except Exception as e:
                        if "already exists" in str(e).lower():
                            logger.info(f"Index already exists, skipping")
                        else:
                            logger.warning(f"Failed to create index: {e}")
                
                logger.info("✅ Security migration completed successfully")
                secure_logger.log_system_event("migration_complete", "Security database migration completed successfully")
                
            except Exception as e:
                logger.error(f"Migration failed: {e}")
                secure_logger.log_security_event("migration_failed", f"Security migration failed: {str(e)}", severity="error")
                raise e
                
    except Exception as e:
        logger.error(f"Security migration failed: {e}")
        secure_logger.log_security_event("migration_error", f"Security migration error: {str(e)}", severity="critical")
        return False
    
    return True

async def verify_migration():
    """
    Verify that the migration was successful
    """
    try:
        logger.info("Verifying security migration...")
        
        async with async_engine.connect() as connection:
            # Check if security_logs table exists
            result = await connection.execute(text(
                "SELECT table_name FROM information_schema.tables WHERE table_name = 'security_logs'"
            ))
            
            if result.fetchone():
                logger.info("✅ Security logs table verified")
            else:
                logger.error("❌ Security logs table not found")
                return False
            
            # Check if new columns exist in users table
            result = await connection.execute(text(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'access_level'"
            ))
            
            if result.fetchone():
                logger.info("✅ User security columns verified")
            else:
                logger.error("❌ User security columns not found")
                return False
        
        logger.info("✅ Security migration verification completed")
        secure_logger.log_system_event("migration_verified", "Security migration verification completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Migration verification failed: {e}")
        secure_logger.log_security_event("migration_verification_failed", f"Migration verification failed: {str(e)}", severity="error")
        return False

async def main():
    logger.info("Running security migration script...")
    
    if await run_security_migration():
        if await verify_migration():
            logger.info("🔒 Security migration completed and verified successfully!")
        else:
            logger.error("Security migration verification failed")
    else:
        logger.error("Security migration failed")

if __name__ == "__main__":
    asyncio.run(main())