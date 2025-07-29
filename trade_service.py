from sqlalchemy.future import select
from db import AsyncSessionLocal
from models import Trade, User
from datetime import datetime
from typing import List, Dict, Optional
import logging
import os

logger = logging.getLogger("TradeService")

# Check if database is skipped
SKIP_DATABASE = os.getenv("SKIP_DATABASE", "false").lower() == "true"

class TradeService:
    """Async service for managing trades in the database"""
    async def create_trade(self, telegram_user_id: int, symbol: str, action: str, quantity: float, price: float) -> Dict:
        if SKIP_DATABASE:
            # Simulate trade creation when database is skipped
            import random
            trade_id = random.randint(1000, 9999)
            logger.info(f"[SIMULATED] Trade created for user {telegram_user_id}: {action} {quantity} {symbol} @ {price}")
            return {"success": True, "trade_id": trade_id, "simulated": True}
        
        try:
            async with AsyncSessionLocal() as session:
                # Ensure user exists
                user = await session.execute(select(User).where(User.telegram_id == str(telegram_user_id)))
                user_obj = user.scalars().first()
                if not user_obj:
                    user_obj = User(telegram_id=str(telegram_user_id))
                    session.add(user_obj)
                    await session.commit()
                    await session.refresh(user_obj)
                trade = Trade(
                    user_id=user_obj.id,
                    symbol=symbol.upper(),
                    action=action,
                    quantity=quantity,
                    price=price,
                    executed_at=datetime.utcnow()
                )
                session.add(trade)
                await session.commit()
                await session.refresh(trade)
                logger.info(f"Trade created for user {telegram_user_id}: {action} {quantity} {symbol} @ {price}")
                return {"success": True, "trade_id": trade.id}
        except Exception as e:
            logger.error(f"Error creating trade: {e}")
            return {"success": False, "error": str(e)}

    async def list_trades(self, telegram_user_id: int) -> List[Dict]:
        if SKIP_DATABASE:
            # Return empty list when database is skipped
            logger.info(f"[SIMULATED] Listing trades for user {telegram_user_id} - no database available")
            return []
        
        try:
            async with AsyncSessionLocal() as session:
                user = await session.execute(select(User).where(User.telegram_id == str(telegram_user_id)))
                user_obj = user.scalars().first()
                if not user_obj:
                    return []
                trades = await session.execute(select(Trade).where(Trade.user_id == user_obj.id))
                trade_objs = trades.scalars().all()
                return [
                    {
                        "id": trade.id,
                        "symbol": trade.symbol,
                        "action": trade.action,
                        "quantity": trade.quantity,
                        "price": trade.price,
                        "executed_at": trade.executed_at,
                    }
                    for trade in trade_objs
                ]
        except Exception as e:
            logger.error(f"Error listing trades: {e}")
            return []

    async def delete_trade(self, telegram_user_id: int, trade_id: int) -> Dict:
        if SKIP_DATABASE:
            # Simulate trade deletion when database is skipped
            logger.info(f"[SIMULATED] Trade {trade_id} deleted for user {telegram_user_id}")
            return {"success": True, "simulated": True}
        
        try:
            async with AsyncSessionLocal() as session:
                user = await session.execute(select(User).where(User.telegram_id == str(telegram_user_id)))
                user_obj = user.scalars().first()
                if not user_obj:
                    return {"success": False, "error": "User not found"}
                trade = await session.execute(select(Trade).where(Trade.id == trade_id, Trade.user_id == user_obj.id))
                trade_obj = trade.scalars().first()
                if not trade_obj:
                    return {"success": False, "error": "Trade not found"}
                await session.delete(trade_obj)
                await session.commit()
                logger.info(f"Trade {trade_id} deleted for user {telegram_user_id}")
                return {"success": True}
        except Exception as e:
            logger.error(f"Error deleting trade: {e}")
            return {"success": False, "error": str(e)} 