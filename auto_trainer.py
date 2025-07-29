"""
Auto-Training Service for Qlib Models
Handles scheduled model retraining and signal updates
"""
import asyncio
import schedule
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from logger import logger
from qlib_service import QlibService

class AutoTrainer:
    """Service for automatic Qlib model training and signal updates"""
    
    def __init__(self, qlib_service: QlibService):
        self.qlib_service = qlib_service
        self.is_running = False
        self.last_training = None
        self.training_schedule = "daily"  # daily, weekly, monthly
        self.admin_notifications = True
        
    async def start_auto_training(self):
        """Start the auto-training scheduler"""
        if self.is_running:
            logger.warning("Auto-trainer is already running")
            return
            
        self.is_running = True
        logger.info("Starting auto-training scheduler...")
        
        # Schedule daily training at 6 AM
        schedule.every().day.at("06:00").do(self._train_model_task)
        
        # Schedule weekly training on Sundays at 6 AM
        schedule.every().sunday.at("06:00").do(self._train_model_weekly_task)
        
        # Run the scheduler
        while self.is_running:
            schedule.run_pending()
            await asyncio.sleep(60)  # Check every minute
            
    def stop_auto_training(self):
        """Stop the auto-training scheduler"""
        self.is_running = False
        logger.info("Auto-training scheduler stopped")
        
    async def _train_model_task(self):
        """Daily model training task"""
        try:
            logger.info("Starting daily model training...")
            start_time = datetime.utcnow()
            
            # Train the model
            signals = self.qlib_service.train_basic_model()
            
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            self.last_training = end_time
            
            # Log training results
            logger.info(f"Daily training completed in {duration:.2f} seconds")
            logger.info(f"Generated {len(signals)} signals")
            
            # Send admin notification if enabled
            if self.admin_notifications:
                await self._send_admin_notification(
                    f"✅ Daily Qlib training completed\n"
                    f"📊 Generated {len(signals)} signals\n"
                    f"⏱️ Duration: {duration:.2f}s\n"
                    f"🕐 Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}"
                )
                
        except Exception as e:
            logger.error(f"Daily training failed: {e}")
            if self.admin_notifications:
                await self._send_admin_notification(f"❌ Daily training failed: {str(e)}")
                
    async def _train_model_weekly_task(self):
        """Weekly model training task (more comprehensive)"""
        try:
            logger.info("Starting weekly model training...")
            start_time = datetime.utcnow()
            
            # Train with more data for weekly training
            signals = self.qlib_service.train_basic_model(
                start_date="2015-01-01",  # More historical data
                end_date=datetime.utcnow().strftime('%Y-%m-%d')
            )
            
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            self.last_training = end_time
            
            logger.info(f"Weekly training completed in {duration:.2f} seconds")
            logger.info(f"Generated {len(signals)} signals")
            
            if self.admin_notifications:
                await self._send_admin_notification(
                    f"🔄 Weekly Qlib training completed\n"
                    f"📊 Generated {len(signals)} signals\n"
                    f"⏱️ Duration: {duration:.2f}s\n"
                    f"🕐 Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}"
                )
                
        except Exception as e:
            logger.error(f"Weekly training failed: {e}")
            if self.admin_notifications:
                await self._send_admin_notification(f"❌ Weekly training failed: {str(e)}")
                
    async def _send_admin_notification(self, message: str):
        """Send notification to admin (placeholder for Telegram integration)"""
        # This will be integrated with the Telegram bot
        logger.info(f"Admin notification: {message}")
        
    def get_training_status(self) -> Dict:
        """Get current training status"""
        return {
            'is_running': self.is_running,
            'last_training': self.last_training.isoformat() if self.last_training else None,
            'schedule': self.training_schedule,
            'admin_notifications': self.admin_notifications
        }
        
    async def manual_train(self):
        """Trigger manual training"""
        await self._train_model_task()