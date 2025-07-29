#!/usr/bin/env python3
"""
Memory System Database Migration Script

This script creates the necessary database tables for the intelligent memory system.
Run this script to migrate your existing database to support the new memory features.
"""

import os
import sys
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import Base
from intelligent_memory_system import UserMemory, MemoryInsight
from db import DATABASE_URL

def run_migration():
    """
    Run the database migration to add memory system tables.
    """
    print("🚀 Starting Memory System Database Migration...")
    
    try:
        # Convert async URL to sync URL for migration
        sync_url = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
        
        # Create database engine
        engine = create_engine(sync_url, connect_args={"sslmode": "require"})
        
        # Create all tables (this will only create new ones)
        print("📊 Creating memory system tables...")
        Base.metadata.create_all(engine, tables=[
            UserMemory.__table__,
            MemoryInsight.__table__
        ])
        
        # Verify tables were created
        Session = sessionmaker(bind=engine)
        session = Session()
        
        try:
            # Test if tables exist by running simple queries
            session.execute(text("SELECT COUNT(*) FROM user_memories LIMIT 1"))
            session.execute(text("SELECT COUNT(*) FROM memory_insights LIMIT 1"))
            session.commit()
            
            print("✅ Memory system tables created successfully!")
            print("📋 Created tables:")
            print("   - user_memories")
            print("   - memory_insights")
            
        except Exception as e:
            print(f"❌ Error verifying tables: {e}")
            session.rollback()
            return False
        finally:
            session.close()
            
        print("🎉 Migration completed successfully!")
        print("\n📝 Next steps:")
        print("1. Restart your bot to use the new memory system")
        print("2. The memory system will automatically start learning from user interactions")
        print("3. Check logs for memory system activity")
        
        return True
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Check your DATABASE_URL in config.py")
        print("2. Ensure the database server is running")
        print("3. Verify database permissions")
        return False

def check_migration_status():
    """Check if the memory system tables exist"""
    try:
        # Convert async URL to sync URL for checking
        sync_url = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
        
        # Create sync engine
        engine = create_engine(sync_url, connect_args={"sslmode": "require"})
        
        with engine.begin() as conn:
            # Check if user_memories table exists
            result = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'user_memories'
                )
            """))
            user_memories_exists = result.scalar()
            
            # Check if memory_insights table exists
            result = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'memory_insights'
                )
            """))
            memory_insights_exists = result.scalar()
            
            return user_memories_exists and memory_insights_exists
        
    except Exception as e:
        print(f"❌ Error checking migration status: {e}")
        return False

def main():
    """
    Main migration function.
    """
    print("🧠 TradeAI Companion - Memory System Migration")
    print("=" * 50)
    
    # Check if migration is needed
    if check_migration_status():
        print("✅ Memory system tables already exist!")
        print("No migration needed.")
        return
    
    print("📋 This will create the following tables:")
    print("   - user_memories: Stores user interaction memories")
    print("   - memory_insights: Stores learned user patterns")
    print()
    
    # Confirm migration
    response = input("Do you want to proceed with the migration? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("Migration cancelled.")
        return
    
    # Run migration
    success = run_migration()
    
    if success:
        print("\n🎯 Migration Summary:")
        print(f"   - Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("   - Status: SUCCESS")
        print("   - Memory system is now ready!")
    else:
        print("\n❌ Migration failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()