"""Real-Time Alert Service
Handles price alerts and notifications with high-traffic optimizations
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Callable, Awaitable
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import update as sqlalchemy_update
from models import User, Alert
from db import AsyncSessionLocal
from logger import logger
from market_data_service import MarketDataService
from performance_cache import (
    performance_cache, response_cache, connection_pool,
    cache_result, with_connection_pool
)

class AlertService:
    """Service for managing real-time price alerts with high-traffic optimizations"""
    def __init__(self, market_service: MarketDataService, notification_callback: Optional[Callable[[int, str], Awaitable[None]]] = None):
        self.market_service = market_service
        self.notification_callback = notification_callback
        self.is_running = False
        self.check_interval = 30  # Reduced to 30 seconds for better responsiveness
        self.batch_size = 50  # Process alerts in batches
        self.price_cache_ttl = 30  # Cache prices for 30 seconds
        self.user_cache_ttl = 300  # Cache user data for 5 minutes

    async def add_alert(self, telegram_user_id: int, symbol: str, condition: str, target_price: float, alert_type: str = "price") -> Dict:
        """Add a new price alert to the database with caching optimization"""
        try:
            # Cache user lookup
            user_cache_key = f"user_{telegram_user_id}"
            user_obj = performance_cache.get(user_cache_key)
            
            async with AsyncSessionLocal() as session:
                if not user_obj:
                    # Ensure user exists
                    user = await session.execute(select(User).where(User.telegram_id == str(telegram_user_id)))
                    user_obj = user.scalars().first()
                    if not user_obj:
                        user_obj = User(telegram_id=str(telegram_user_id))
                        session.add(user_obj)
                        await session.commit()
                        await session.refresh(user_obj)
                    
                    # Cache user object
                    performance_cache.set(user_cache_key, user_obj, ttl=self.user_cache_ttl)
                
                alert = Alert(
                    user_id=user_obj.id,
                    symbol=symbol.upper(),
                    condition=f"{condition} {target_price:.2f}",
                    is_active=True,
                    created_at=datetime.utcnow()
                )
                session.add(alert)
                await session.commit()
                await session.refresh(alert)
                
                # Invalidate user alerts cache
                alerts_cache_key = f"user_alerts_{telegram_user_id}"
                performance_cache.delete(alerts_cache_key)
                
                logger.info(f"DB Alert added for user {telegram_user_id}: {symbol} {condition} {target_price}")
                return {'success': True, 'alert_id': alert.id, 'message': 'Alert created successfully'}
        except Exception as e:
            logger.error(f"Error adding alert: {e}")
            return {'success': False, 'error': str(e)}

    async def remove_alert(self, telegram_user_id: int, alert_id: int) -> Dict:
        """Remove an alert from the database"""
        try:
            async with AsyncSessionLocal() as session:
                # Check user cache first
                user_cache_key = f"user_{telegram_user_id}"
                cached_user = performance_cache.get(user_cache_key)
                
                if cached_user is None:
                    user = await session.execute(select(User).where(User.telegram_id == str(telegram_user_id)))
                    user_obj = user.scalars().first()
                    if not user_obj:
                        return {'success': False, 'error': 'User not found'}
                    # Cache user object
                    performance_cache.set(user_cache_key, user_obj, ttl=self.user_cache_ttl)
                    cached_user = user_obj
                else:
                    user_obj = cached_user
                
                alert = await session.execute(select(Alert).where(Alert.id == alert_id, Alert.user_id == user_obj.id))
                alert_obj = alert.scalars().first()
                if not alert_obj:
                    return {'success': False, 'error': 'Alert not found'}
                await session.delete(alert_obj)
                await session.commit()
                
                # Force invalidate ALL related cache entries
                alerts_cache_key = f"user_alerts_{telegram_user_id}"
                performance_cache.delete(alerts_cache_key)
                
                # Clear cache entries created by @cache_result decorator
                # The decorator creates keys like: get_user_alerts_{hash_of_args}
                
                # Generate the same cache key that @cache_result decorator would create
                # This must match exactly what the decorator generates
                decorator_cache_key = f"get_user_alerts_{performance_cache._generate_key(telegram_user_id)}"
                
                # Also clear any other cached function results that might contain alert data
                cache_keys_to_clear = [
                    f"user_alerts_{telegram_user_id}",
                    f"alerts_command_{telegram_user_id}",
                    f"get_user_alerts_{telegram_user_id}",
                    decorator_cache_key
                ]
                
                # Clear all cache keys and log what we're clearing
                for cache_key in cache_keys_to_clear:
                    deleted = performance_cache.delete(cache_key)
                    logger.debug(f"Cache key '{cache_key}': {'deleted' if deleted else 'not found'}")
                
                # Also try to find and clear any cache keys that start with get_user_alerts_
                # This is a more aggressive approach to ensure we clear the decorator cache
                with performance_cache.lock:
                    keys_to_delete = []
                    for key in performance_cache.cache.keys():
                        if key.startswith('get_user_alerts_'):
                            keys_to_delete.append(key)
                    
                    for key in keys_to_delete:
                        performance_cache.delete(key)
                        logger.debug(f"Aggressively cleared cache key: {key}")
                
                logger.info(f"DB Alert {alert_id} removed for user {telegram_user_id}, cache cleared")
                return {'success': True, 'message': 'Alert removed successfully'}
        except Exception as e:
            logger.error(f"Error removing alert: {e}")
            return {'success': False, 'error': str(e)}

    @cache_result(ttl=60)  # Cache user alerts for 1 minute
    async def get_user_alerts(self, telegram_user_id: int) -> List[Dict]:
        """Get all alerts for a user from the database with caching"""
        try:
            # Check cache first
            alerts_cache_key = f"user_alerts_{telegram_user_id}"
            cached_alerts = performance_cache.get(alerts_cache_key)
            if cached_alerts is not None:
                return cached_alerts
            
            # Get user from cache or database
            user_cache_key = f"user_{telegram_user_id}"
            user_obj = performance_cache.get(user_cache_key)
            
            async with AsyncSessionLocal() as session:
                if not user_obj:
                    user = await session.execute(select(User).where(User.telegram_id == str(telegram_user_id)))
                    user_obj = user.scalars().first()
                    if not user_obj:
                        return []
                    # Cache user object
                    performance_cache.set(user_cache_key, user_obj, ttl=self.user_cache_ttl)
                
                alerts = await session.execute(select(Alert).where(Alert.user_id == user_obj.id))
                alert_objs = alerts.scalars().all()
                result = [
                    {
                        'id': alert.id,
                        'symbol': alert.symbol,
                        'condition': alert.condition,
                        'is_active': alert.is_active,
                        'created_at': alert.created_at,
                    }
                    for alert in alert_objs
                ]
                
                # Cache the result
                performance_cache.set(alerts_cache_key, result, ttl=60)
                return result
                
        except Exception as e:
            logger.error(f"Error fetching user alerts: {e}")
            return []

    async def get_all_alerts(self) -> List[Alert]:
        """Get all alerts (for admin) from the database"""
        try:
            async with AsyncSessionLocal() as session:
                alerts = await session.execute(select(Alert))
                return alerts.scalars().all()
        except Exception as e:
            logger.error(f"Error fetching all alerts: {e}")
            return []

    async def start_alert_monitoring(self):
        """Start monitoring alerts from the database"""
        if self.is_running:
            logger.warning("Alert monitoring is already running")
            return
        self.is_running = True
        logger.info("Starting DB alert monitoring...")
        while self.is_running:
            try:
                await self._check_alerts()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in alert monitoring: {e}")
                await asyncio.sleep(self.check_interval)

    def stop_alert_monitoring(self):
        """Stop monitoring alerts"""
        self.is_running = False
        logger.info("Alert monitoring stopped")

    async def _check_alerts(self):
        """Check all active alerts from the database with batch processing"""
        try:
            async with AsyncSessionLocal() as session:
                alerts = await session.execute(select(Alert).where(Alert.is_active == True))
                alert_objs = alerts.scalars().all()
                
                # Process alerts in batches for better performance
                for i in range(0, len(alert_objs), self.batch_size):
                    batch = alert_objs[i:i + self.batch_size]
                    
                    # Group alerts by symbol to minimize API calls
                    symbol_groups = {}
                    for alert in batch:
                        symbol = alert.symbol
                        if symbol not in symbol_groups:
                            symbol_groups[symbol] = []
                        symbol_groups[symbol].append(alert)
                    
                    # Process each symbol group
                    await self._process_symbol_groups(symbol_groups, session)
                    
                    # Small delay between batches to prevent overwhelming the system
                    if i + self.batch_size < len(alert_objs):
                        await asyncio.sleep(0.1)
                        
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")

    async def _process_symbol_groups(self, symbol_groups: Dict[str, List], session):
        """Process alerts grouped by symbol for efficient price fetching"""
        for symbol, alerts in symbol_groups.items():
            try:
                # Check price cache first
                price_cache_key = f"price_{symbol}"
                cached_price_data = performance_cache.get(price_cache_key)
                
                if cached_price_data is None:
                    # Get current price from market service
                    price_data = await self.market_service.get_stock_price(symbol, alerts[0].user_id)
                    if not price_data or 'price' not in price_data:
                        logger.warning(f"No price data available for {symbol}")
                        continue
                    
                    # Cache the price data
                    performance_cache.set(price_cache_key, price_data, ttl=self.price_cache_ttl)
                    cached_price_data = price_data
                
                current_price = float(cached_price_data['price'])
                
                # Check all alerts for this symbol
                for alert in alerts:
                    await self._check_single_alert_with_price(alert, current_price, session)
                    
            except Exception as e:
                logger.error(f"Error processing symbol group {symbol}: {e}")
    
    async def _check_single_alert_with_price(self, alert: Alert, current_price: float, session):
        """Check a single alert with pre-fetched price data"""
        try:
            symbol = alert.symbol
            condition = alert.condition
            triggered = False
            message = ""
            
            # Parse condition (e.g., "above 150.00" or "below 100.50")
            try:
                parts = condition.lower().split()
                if len(parts) >= 2:
                    direction = parts[0]  # "above" or "below"
                    target_price = float(parts[1])
                    
                    if direction == "above" and current_price > target_price:
                        triggered = True
                        message = f"🚨 **Alert Triggered!** 🚨\n\n📈 {symbol} is now **${current_price:.2f}**\n🎯 Target: Above ${target_price:.2f}\n✅ Condition met!"
                    elif direction == "below" and current_price < target_price:
                        triggered = True
                        message = f"🚨 **Alert Triggered!** 🚨\n\n📉 {symbol} is now **${current_price:.2f}**\n🎯 Target: Below ${target_price:.2f}\n✅ Condition met!"
                        
                    logger.debug(f"Checking alert {alert.id}: {symbol} ${current_price:.2f} {condition} (triggered: {triggered})")
                elif len(parts) == 1:
                    # Handle incomplete conditions - deactivate them
                    logger.warning(f"Incomplete condition format for alert {alert.id}: '{condition}' - deactivating alert")
                    alert.is_active = False
                    await session.commit()
                    return
                else:
                    logger.error(f"Invalid condition format for alert {alert.id}: {condition}")
                    return
                    
            except (ValueError, IndexError) as e:
                logger.error(f"Error parsing condition '{condition}' for alert {alert.id}: {e}")
                return
            
            if triggered:
                # Deactivate the alert
                alert.is_active = False
                await session.commit()
                
                # Invalidate user alerts cache
                user = await session.execute(select(User).where(User.id == alert.user_id))
                user_obj = user.scalars().first()
                if user_obj:
                    alerts_cache_key = f"user_alerts_{user_obj.telegram_id}"
                    performance_cache.delete(alerts_cache_key)
                
                # Send notification
                await self._send_alert_notification(alert, message)
                logger.info(f"DB Alert triggered: {alert.id} - {symbol} {condition} at ${current_price:.2f}")
                
        except Exception as e:
            logger.error(f"Error checking alert {getattr(alert, 'id', 'unknown')}: {e}")

    async def _send_alert_notification(self, alert: Alert, message: str):
        """Send alert notification"""
        try:
            if self.notification_callback:
                # Get the telegram_id from the user table
                async with AsyncSessionLocal() as session:
                    user = await session.execute(select(User).where(User.id == alert.user_id))
                    user_obj = user.scalars().first()
                    if user_obj:
                        telegram_id = int(user_obj.telegram_id)
                        # Simply await the callback - it should be properly bound now
                        await self.notification_callback(telegram_id, message)
                        logger.info(f"Alert notification sent to user {telegram_id}")
                    else:
                        logger.error(f"User not found for alert {alert.id}")
            else:
                logger.info(f"Alert notification for user {alert.user_id}: {message}")
        except Exception as e:
            logger.error(f"Error sending alert notification: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")

    async def get_alert_stats(self) -> Dict:
        """Get alert system statistics with caching"""
        try:
            # Check cache first
            stats_cache_key = "alert_system_stats"
            cached_stats = performance_cache.get(stats_cache_key)
            
            if cached_stats is None:
                async with AsyncSessionLocal() as session:
                    total_alerts = await session.execute(select(Alert))
                    total_alerts_count = len(total_alerts.scalars().all())
                    active_alerts = await session.execute(select(Alert).where(Alert.is_active == True))
                    active_alerts_count = len(active_alerts.scalars().all())
                    triggered_alerts = await session.execute(select(Alert).where(Alert.is_active == False))
                    triggered_alerts_count = len(triggered_alerts.scalars().all())
                    total_users = await session.execute(select(User))
                    total_users_count = len(total_users.scalars().all())
                    
                    stats = {
                        'total_alerts': total_alerts_count,
                        'active_alerts': active_alerts_count,
                        'triggered_alerts': triggered_alerts_count,
                        'total_users': total_users_count,
                        'is_running': self.is_running,
                        'cache_stats': {
                            'price_cache_hits': getattr(performance_cache, 'hit_count', 0),
                            'price_cache_misses': getattr(performance_cache, 'miss_count', 0)
                        }
                    }
                    
                    # Cache stats for 30 seconds
                    performance_cache.set(stats_cache_key, stats, ttl=30)
                    return stats
            else:
                # Update monitoring status in cached stats
                cached_stats['is_running'] = self.is_running
                return cached_stats
                
        except Exception as e:
            logger.error(f"Error getting alert stats: {e}")
            return {
                'total_alerts': 0,
                'active_alerts': 0,
                'triggered_alerts': 0,
                'total_users': 0,
                'is_running': self.is_running,
                'cache_stats': {
                    'price_cache_hits': 0,
                    'price_cache_misses': 0
                }
            }