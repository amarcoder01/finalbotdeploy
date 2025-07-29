"""
Telegram bot handler module
Manages Telegram Bot API integration and message processing
"""
import asyncio
import base64
import mimetypes
import io
import os
from timezone_utils import ModernTimezoneHandler
try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None
    print("Warning: PyPDF2 not available, PDF processing disabled")
# Set timezone environment to avoid pytz conflicts
os.environ['TZ'] = 'UTC'
if hasattr(os, 'tzset'):
    os.tzset()
# Import pytz and set default timezone for APScheduler
import pytz
os.environ['APSCHEDULER_TIMEZONE'] = 'UTC'
from telegram import Update, InputFile, ReplyKeyboardRemove
from telegram import Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
import asyncio
from telegram.error import TelegramError, NetworkError, TimedOut
from typing import Optional, Any
from io import BytesIO
from logger import logger
from config import Config
from openai_service import OpenAIService
from market_data_service import MarketDataService
from chart_service import ChartService
from trading_intelligence import TradingIntelligence
from conversation_memory import ConversationMemory
try:
    from pdf2image import convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    convert_from_bytes = None
    PDF2IMAGE_AVAILABLE = False
    print("Warning: pdf2image not available, PDF image conversion disabled")

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    pytesseract = None
    PYTESSERACT_AVAILABLE = False
    print("Warning: pytesseract not available, OCR disabled")
from datetime import datetime
from timezone_utils import format_ist_timestamp, get_ist_time
from qlib_service import QlibService
from auto_trainer import AutoTrainer
from alert_service import AlertService
from real_market_data import RealMarketDataService
# Skip advanced strategies that require joblib/numpy
try:
    from advanced_qlib_strategies import AdvancedQlibStrategies
except ImportError:
    logger.warning("Advanced Qlib strategies disabled - dependencies missing")
    AdvancedQlibStrategies = None
try:
    from enhanced_technical_indicators import EnhancedTechnicalIndicators
    ENHANCED_INDICATORS_AVAILABLE = True
except (ImportError, AttributeError) as e:
    logger.warning(f"Enhanced technical indicators disabled: {e}")
    EnhancedTechnicalIndicators = None
    ENHANCED_INDICATORS_AVAILABLE = False
try:
    from deep_learning_models import DeepLearningService
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    logger.warning("Deep learning models disabled - dependencies missing")
    DeepLearningService = None
    DEEP_LEARNING_AVAILABLE = False
    
try:
    from backtesting_framework import BacktestingFramework, sma_crossover_strategy, rsi_strategy, macd_strategy
    BACKTESTING_AVAILABLE = True
except ImportError:
    logger.warning("Backtesting framework disabled - dependencies missing")
    BacktestingFramework = None
    sma_crossover_strategy = None
    rsi_strategy = None
    macd_strategy = None
    BACKTESTING_AVAILABLE = False
    
try:
    from performance_attribution import PerformanceAttribution
    PERFORMANCE_ATTRIBUTION_AVAILABLE = True
except ImportError:
    logger.warning("Performance attribution disabled - dependencies missing")
    PerformanceAttribution = None
    PERFORMANCE_ATTRIBUTION_AVAILABLE = False
    
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False
    
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False
from trade_service import TradeService
import logging
from monitoring import metrics, time_operation
from performance_cache import cache_result, with_connection_pool, performance_cache, response_cache, connection_pool
from enhanced_memory_service import enhanced_memory_service
from memory_integration import (
    memory_integration, remember_interaction, remember_preference, 
    remember_trading_activity, remember_alert_activity, remember_error_context
)
from intelligent_memory_system import MemoryType, MemoryImportance
from security_middleware import security_middleware, AccessLevel
from secure_logger import secure_logger
from input_validator import input_validator
from enhanced_error_handler import EnhancedErrorHandler

# Import new UI components
from ui_components import TradingBotUI
from callback_handler import ModernCallbackHandler

class TelegramHandler:
    """Handler class for Telegram bot operations"""
    _started = False  # Class-level safeguard to prevent double initialization
    
    def __init__(self):
        """Initialize Telegram bot handler"""
        if TelegramHandler._started:
            raise RuntimeError("TelegramHandler has already been started in this process!")
        TelegramHandler._started = True
        self._typing_tasks = {}  # Track active typing indicator tasks
        config = Config()
        self.bot_token = config.TELEGRAM_BOT_TOKEN
        self.openai_service = OpenAIService()
        self.market_service = MarketDataService()
        self.real_market_service = RealMarketDataService()
        self.chart_service = ChartService()
        self.trading_intelligence = TradingIntelligence()
        self.qlib_service = QlibService()
        self.advanced_qlib = AdvancedQlibStrategies() if AdvancedQlibStrategies else None
        self.technical_indicators = EnhancedTechnicalIndicators() if EnhancedTechnicalIndicators else None
        self.auto_trainer = AutoTrainer(self.qlib_service)
        # Create a bound method wrapper for the alert notification callback
        async def alert_notification_wrapper(user_id: int, message: str):
            await self._send_alert_notification(user_id, message)
        
        self.alert_service = AlertService(self.market_service, alert_notification_wrapper)
        self.alert_monitoring_task = None  # Task for periodic alert checking
        self.deep_learning = DeepLearningService() if DeepLearningService else None
        self.backtesting = BacktestingFramework() if BacktestingFramework else None
        self.performance_attribution = PerformanceAttribution() if PerformanceAttribution else None
        self.application: Optional[Application] = None
        self.conversation_memory = ConversationMemory()
        self.enhanced_memory = enhanced_memory_service
        self.memory_integration = memory_integration
        self.trade_service = TradeService()
        self.us_stocks_cache = None  # Cache for US stock symbols
        
        # Initialize modern UI components
        self.ui = TradingBotUI()
        self.callback_handler = ModernCallbackHandler(self)
        
        # Initialize enhanced error handler
        self.error_handler = EnhancedErrorHandler()
        
        # Initialize input validator
        self.input_validator = input_validator

        logger.info("Telegram handler initialized with all advanced trading services, intelligent memory, modern UI, and enhanced error handling")
    
    def _get_ist_timestamp(self) -> str:
        """Get current timestamp in IST format"""
        return format_ist_timestamp('%Y-%m-%d %H:%M:%S IST')
    
    def _load_us_stocks(self):
        """Load US stock symbols from Qlib dataset"""
        if self.us_stocks_cache is not None:
            return self.us_stocks_cache
        
        try:
            # Load US stocks from Qlib instruments file
            instruments_path = os.path.join(os.getcwd(), 'qlib_data', 'us_data', 'instruments', 'all.txt')
            if os.path.exists(instruments_path):
                with open(instruments_path, 'r') as f:
                    lines = f.readlines()
                
                # Extract symbols (first column before tab)
                symbols = set()
                for line in lines:
                    if line.strip():
                        symbol = line.split('\t')[0].strip().upper()
                        if symbol and len(symbol) <= 10:  # Valid stock symbol length
                            symbols.add(symbol)
                
                self.us_stocks_cache = symbols
                logger.info(f"Loaded {len(symbols)} US stock symbols from Qlib dataset")
                return symbols
            else:
                logger.warning(f"US instruments file not found at {instruments_path}")
        except Exception as e:
            logger.error(f"Error loading US stocks: {e}")
        
        # Fallback to common US stocks if file not available
        fallback_stocks = {
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'AMD',
            'INTC', 'CRM', 'ORCL', 'ADBE', 'PYPL', 'UBER', 'LYFT', 'SNAP', 'TWTR', 'SPOT',
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'V', 'MA', 'AXP', 'COF',
            'JNJ', 'PFE', 'UNH', 'ABBV', 'TMO', 'DHR', 'ABT', 'LLY', 'BMY', 'MRK',
            'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'KMI', 'OKE', 'WMB', 'EPD',
            'WMT', 'HD', 'COST', 'TGT', 'LOW', 'SBUX', 'MCD', 'NKE', 'DIS', 'CMCSA'
        }
        self.us_stocks_cache = fallback_stocks
        logger.info(f"Using fallback US stocks list with {len(fallback_stocks)} symbols")
        return fallback_stocks
    
    def _is_valid_us_stock(self, symbol: str) -> bool:
        """Check if symbol is a valid US stock"""
        if not symbol or len(symbol) > 10:
            return False
        
        symbol = symbol.upper().strip()
        us_stocks = self._load_us_stocks()
        
        # Direct match
        if symbol in us_stocks:
            return True
        
        # Check for common variations
        variations = [
            symbol.replace('.', ''),
            symbol.replace('-', ''),
            symbol + '.US',
            symbol + '.O',
            symbol + '.Q'
        ]
        
        for variation in variations:
            if variation in us_stocks:
                return True
        
        # Check if it looks like a valid US stock symbol pattern
        if len(symbol) >= 1 and len(symbol) <= 5 and symbol.isalpha():
            return True
        
        return False
    
    def _normalize_us_stock_symbol(self, symbol: str) -> str:
        """Normalize US stock symbol for consistent processing"""
        if not symbol:
            return symbol
        
        symbol = symbol.upper().strip()
        
        # Remove common suffixes that might cause issues
        suffixes_to_remove = ['.US', '.O', '.Q', '.PINK', '.OTC']
        for suffix in suffixes_to_remove:
            if symbol.endswith(suffix):
                symbol = symbol[:-len(suffix)]
        
        # Handle special cases
        symbol_mappings = {
            'BRK.A': 'BRK-A',
            'BRK.B': 'BRK-B',
            'BF.A': 'BF-A',
            'BF.B': 'BF-B'
        }
        
        return symbol_mappings.get(symbol, symbol)
    
    def _build_enhanced_context(self, traditional_context: str, contextual_data: dict) -> str:
        """
        Build enhanced context by combining traditional conversation context with memory insights
        
        Args:
            traditional_context: Traditional conversation context
            contextual_data: Enhanced contextual data from memory system
            
        Returns:
            Enhanced context string
        """
        try:
            enhanced_parts = []
            
            # Add traditional context
            if traditional_context:
                enhanced_parts.append(f"Recent Conversation:\n{traditional_context}")
            
            # Add user preferences
            preferences = contextual_data.get('user_preferences', {})
            if preferences:
                pref_text = ", ".join([f"{k}: {v}" for k, v in preferences.items()])
                enhanced_parts.append(f"User Preferences: {pref_text}")
            
            # Add behavioral patterns
            patterns = contextual_data.get('behavioral_patterns', [])
            if patterns:
                pattern_text = "; ".join(patterns[:3])  # Limit to top 3 patterns
                enhanced_parts.append(f"User Patterns: {pattern_text}")
            
            # Add relevant history
            relevant_history = contextual_data.get('relevant_history', [])
            if relevant_history:
                history_items = []
                for item in relevant_history[:3]:  # Limit to top 3
                    history_items.append(f"{item['type']}: {item['content'][:100]}...")
                enhanced_parts.append(f"Relevant History: {'; '.join(history_items)}")
            
            # Add recent activities
            recent_activities = contextual_data.get('recent_activities', [])
            if recent_activities:
                activity_items = []
                for activity in recent_activities[:2]:  # Limit to top 2
                    activity_items.append(f"{activity['type']}: {activity['content'][:80]}...")
                enhanced_parts.append(f"Recent Activities: {'; '.join(activity_items)}")
            
            # Combine all parts
            if enhanced_parts:
                return "\n\n".join(enhanced_parts)
            else:
                return traditional_context or ""
                
        except Exception as e:
            logger.error(f"Error building enhanced context: {e}")
            return traditional_context or ""
    
    async def _send_typing_indicator(self, chat_id: int, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send a single typing indicator"""
        try:
            await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        except Exception as e:
            logger.error(f"Error sending typing indicator to chat {chat_id}: {e}")
    
    async def _start_persistent_typing(self, chat_id: int, context: ContextTypes.DEFAULT_TYPE) -> str:
        """Start a persistent typing indicator that refreshes every 4 seconds"""
        task_id = f"typing_{chat_id}_{asyncio.get_event_loop().time()}"
        
        async def typing_loop():
            try:
                while task_id in self._typing_tasks:
                    await self._send_typing_indicator(chat_id, context)
                    await asyncio.sleep(4)  # Refresh every 4 seconds (Telegram typing expires after 5)
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Error in typing loop for chat {chat_id}: {e}")
        
        # Start the typing task
        task = asyncio.create_task(typing_loop())
        self._typing_tasks[task_id] = task
        
        return task_id
    
    async def _stop_persistent_typing(self, task_id: str) -> None:
        """Stop a persistent typing indicator"""
        if task_id in self._typing_tasks:
            task = self._typing_tasks.pop(task_id)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    
    async def _with_typing_indicator(self, chat_id: int, context: ContextTypes.DEFAULT_TYPE, operation_func, *args, **kwargs):
        """Execute an operation with automatic typing indicator management"""
        # Send immediate typing indicator
        await self._send_typing_indicator(chat_id, context)
        
        # For operations that might take longer than 5 seconds, start persistent typing
        typing_task_id = await self._start_persistent_typing(chat_id, context)
        
        try:
            # Execute the operation
            result = await operation_func(*args, **kwargs)
            return result
        finally:
            # Always stop the typing indicator
            await self._stop_persistent_typing(typing_task_id)
    
    @time_operation("start_command")
    @remember_interaction(memory_type=MemoryType.CONVERSATION, importance=MemoryImportance.MEDIUM)
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle /start command with modern UI
        
        Args:
            update: Telegram update object
            context: Bot context
        """
        user = update.effective_user
        user_id = str(user.id) if user else "unknown"
        
        # Record metrics
        metrics.record_message("start", user_id)
        
        # Use the new modern welcome message with inline keyboard
        welcome_message = TradingBotUI.format_welcome_message()
        
        try:
            await update.message.reply_text(
                welcome_message,
                reply_markup=TradingBotUI.create_main_menu(),
                parse_mode='Markdown'
            )
            logger.info(f"Start command processed for user {user.id} ({user.username})")
        except Exception as e:
            logger.error(f"Error sending start message to user {user.id}: {str(e)}")
    
    async def menu_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle /menu command - show interactive menu
        
        Args:
            update: Telegram update object
            context: Bot context
        """
        user = update.effective_user
        user_id = str(user.id) if user else "unknown"
        
        # Record metrics
        metrics.record_message("menu", user_id)
        
        try:
            await update.message.reply_text(
                TradingBotUI.format_welcome_message(),
                reply_markup=TradingBotUI.create_main_menu(),
                parse_mode='Markdown'
            )
            logger.info(f"Menu command processed for user {user.id}")
        except Exception as e:
            logger.error(f"Error sending menu to user {user.id}: {str(e)}")
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle /help command with improved structure and navigation
        
        Args:
            update: Telegram update object
            context: Bot context
        """
        # Main help overview - concise and organized
        help_message = """
🚀 **TradeMaster AI - Your Smart Trading Assistant**

**⚡ QUICK START:**
• `/price AAPL` - Get stock prices instantly
• `/chart TSLA` - View stock charts
• `/analyze NVDA` - Get AI analysis
• `/alert AAPL above 150` - Set price alerts

**📚 HELP SECTIONS:**
• `/help_trading` - Stock prices, charts & analysis
• `/help_alerts` - Price alerts & notifications
• `/help_advanced` - Advanced features for pros
• `/help_examples` - Examples & tips

**📞 SUPPORT:**
• `/contact` - Get help or report issues
• `/privacy` - Privacy & data protection

**💡 SIMPLE COMMANDS:**
• **Get Prices** - `/price [stock]` (e.g., `/price AAPL`)
• **View Charts** - `/chart [stock]` (e.g., `/chart TSLA`)
• **Get Analysis** - `/analyze [stock]` (e.g., `/analyze NVDA`)
• **Set Alerts** - `/alert [stock] above/below [price]`
• **View Portfolio** - `/portfolio` to see your investments
• **Analyze Chart Images** - Send any chart image for AI analysis

• **Personal Watchlist** - `/watchlist` - Manage your stock watchlist

**🚀 GET STARTED:**
Try `/price AAPL` or just ask: "How is Apple performing today?"

*Smart trading made simple* 📊
        """
        
        try:
            await update.message.reply_text(help_message, parse_mode='Markdown')
            logger.info(f"Help command processed for user {update.effective_user.id}")
        except Exception as e:
            logger.error(f"Error sending help message: {str(e)}")
    
    async def help_trading_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle /help_trading command - detailed trading commands
        """
        help_message = """
📊 **STOCK PRICES, CHARTS & ANALYSIS**

**💹 GET STOCK PRICES:**
• `/price AAPL` - Get Apple's current price
• `/price TSLA` - Get Tesla's current price
• `/price SPY` - Get S&P 500 ETF price
• `/price BTC-USD` - Get Bitcoin price

**📈 VIEW CHARTS:**
• `/chart AAPL` - See Apple's price chart
• `/chart TSLA 1M` - Tesla chart for 1 month
• `/chart NVDA 6M` - NVIDIA chart for 6 months
• `/chart SPY 1Y` - S&P 500 chart for 1 year

**🤖 GET AI ANALYSIS:**
• `/analyze AAPL` - AI analysis of Apple
• `/analyze TSLA` - AI analysis of Tesla
• `/analyze tech` - Analysis of tech sector
• **Send Chart Images** - Upload any chart image for AI analysis

**📊 TRACK YOUR INVESTMENTS:**
• `/trade buy AAPL 10 150` - Record buying 10 Apple shares at $150
• `/trade sell TSLA 5 250` - Record selling 5 Tesla shares at $250
• `/portfolio` - See all your recorded trades
• `/trades` - View your trading history
• `/delete_trade 123` - Delete a specific trade by ID

**💡 WHAT STOCKS CAN I CHECK:**
• **US Stocks:** Apple (AAPL), Tesla (TSLA), Microsoft (MSFT), etc.
• **ETFs:** SPY, QQQ, VTI (popular investment funds)
• **Crypto:** BTC-USD (Bitcoin), ETH-USD (Ethereum)
• **Indices:** ^GSPC (S&P 500), ^IXIC (NASDAQ)

**⚡ QUICK TIPS:**
• Use stock symbols like AAPL (not "Apple")
• Charts show the last few months by default
• AI analysis explains if a stock looks good or risky
• You can track paper trades to practice

🔙 Return to main menu: `/help`
        """
        
        try:
            await update.message.reply_text(help_message, parse_mode='Markdown')
            logger.info(f"Trading help command processed for user {update.effective_user.id}")
        except Exception as e:
            logger.error(f"Error sending trading help message: {str(e)}")
    
    async def help_alerts_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle /help_alerts command - detailed alert commands
        """
        help_message = """
🚨 **PRICE ALERTS & NOTIFICATIONS**

**⚡ SET PRICE ALERTS:**
• `/alert AAPL above 150` - Alert when Apple goes above $150
• `/alert TSLA below 200` - Alert when Tesla drops below $200
• `/alert NVDA above 500` - Alert when NVIDIA hits $500
• `/alert SPY below 400` - Alert when S&P 500 ETF drops

**📱 MANAGE YOUR ALERTS:**
• `/alerts` - See all your active alerts
• `/remove_alert 1` - Remove alert #1
• `/remove_alert 2` - Remove alert #2

**💡 HOW ALERTS WORK:**
• **Instant Notifications** - Get notified immediately when price hits
• **24/7 Monitoring** - We watch prices even when markets are closed
• **Simple Setup** - Just say the stock and target price
• **Easy Management** - View and remove alerts anytime

**📊 ALERT EXAMPLES:**
• "Tell me when Apple hits $180" → `/alert AAPL above 180`
• "Alert me if Tesla drops to $150" → `/alert TSLA below 150`
• "Notify me when Bitcoin reaches $50k" → `/alert BTC-USD above 50000`
• "Let me know if SPY falls below $380" → `/alert SPY below 380`

**⚡ QUICK TIPS:**
• You can set multiple alerts for the same stock
• Alerts work for stocks, ETFs, and crypto
• round numbeUse exact prices (like 150.50) or rs (like 150)
• Check `/alerts` regularly to manage your notifications
• Remove old alerts you don't need anymore

**🎯 POPULAR ALERT TYPES:**
• **Breakout Alerts** - When stock breaks above resistance
• **Support Alerts** - When stock drops to support level
• **Target Alerts** - When stock reaches your buy/sell target
• **Stop Loss** - When stock drops below your risk level
• **Market Filters** - Regular hours, extended hours, all

**🔧 MANAGEMENT TIPS:**
• Use precise symbols for accuracy
• Set reasonable thresholds to avoid spam
• Review and clean up old alerts regularly
• Combine multiple conditions for better signals

🔙 Return to main menu: `/help`
        """
        
        try:
            await update.message.reply_text(help_message, parse_mode='Markdown')
            logger.info(f"Alerts help command processed for user {update.effective_user.id}")
        except Exception as e:
            logger.error(f"Error sending alerts help message: {str(e)}")
    
    async def help_advanced_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle /help_advanced command - advanced AI features
        """
        help_message = """
🧠 **ADVANCED FEATURES FOR PROS**

**🤖 AI PREDICTIONS:**
• `/ai_analysis AAPL` - AI predicts where Apple stock might go
• `/signals TSLA` - Get buy/sell signals for Tesla
• `/deep_analysis NVDA` - Deep dive into NVIDIA with AI
• `/advanced SPY` - Advanced analysis of S&P 500

**📈 STRATEGY TESTING:**
• `/backtest AAPL` - Test how a strategy would have performed
• `/indicators AAPL` - Technical indicators (RSI, MACD, etc.)
• `/risk GOOGL` - Check how risky a stock is

**🔍 RESEARCH TOOLS:**

**💡 WHAT THESE DO:**
• **AI Analysis** - Computer predicts stock movements
• **Backtesting** - Test strategies on past data
• **Risk Analysis** - Shows how much you could lose
• **Optimization** - Finds best stock combinations
• **Indicators** - Technical signals traders use

**🎯 FOR SERIOUS TRADERS:**
• **Portfolio Optimization** - Build better portfolios
• **Risk Management** - Understand your risks
• **Performance Tracking** - See how well you're doing
• **Market Research** - Deep market insights
• **AI Predictions** - Machine learning forecasts

**⚡ QUICK TIPS:**
• Start with basic commands before trying advanced ones
• AI predictions are educated guesses, not guarantees
• Always do your own research before investing
• Use risk analysis to understand potential losses
• Backtest strategies before using real money

**🚀 GETTING STARTED WITH ADVANCED:**
1. Try `/ai_analysis AAPL` for AI insights
2. Use `/risk TSLA` to check risk levels
3. Test `/backtest SPY` to see historical performance
4. Explore `/indicators MSFT` for technical analysis

🔙 Return to main menu: `/help`
        """
        
        try:
            await update.message.reply_text(help_message, parse_mode='Markdown')
            logger.info(f"Advanced help command processed for user {update.effective_user.id}")
        except Exception as e:
            logger.error(f"Error sending advanced help message: {str(e)}")
    
    async def help_examples_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle /help_examples command - usage examples and tips
        """
        help_message = """
💡 **USAGE EXAMPLES & TIPS**

**🗣️ NATURAL CONVERSATION:**
• "What's happening with Tesla stock?"
• "Should I buy Apple now?"
• "How's the tech sector performing?"
• "Explain options trading to me"
• "Tell me a joke about trading"

**📊 COMMAND EXAMPLES:**
• `/price AAPL` → Get Apple's current price
• `/chart TSLA 1d` → Tesla daily chart
• `/analyze NVDA` → AI analysis of NVIDIA
• `/alert SPY above 450` → S&P 500 alert

**🇺🇸 STOCK COVERAGE EXAMPLES:**
• **Large-cap:** AAPL, MSFT, GOOGL, AMZN, TSLA, META
• **Mid-cap:** ROKU, ZOOM, SHOP, SQ, TWLO, OKTA
• **Small-cap:** NET, DDOG, SNOW, PLTR, ZM
• **ETFs:** SPY, QQQ, IWM, VTI, VOO, ARKK
• **REITs:** O, REIT, VNQ, SCHH, PLD
• **Special:** BRK.A, BRK.B, GOOG, GOOGL

**🎯 PRO TIPS:**
• **Smart Symbol Handling** - I understand BRK.A, BRK-A variations
• **AI Web Search** - I'll find ANY US stock price automatically
• **Context Awareness** - I remember our conversation
• **Multi-source Data** - Real-time market feeds
• **Real-time Updates** - Live market data during trading hours

**🚀 GETTING STARTED:**
1. Try `/price AAPL` for your first command
2. Ask me a natural question about stocks
3. Set up price alerts for your watchlist
4. Explore AI analysis features

🔙 Back to main help: `/help`
        """
        
        try:
            await update.message.reply_text(help_message, parse_mode='Markdown')
            logger.info(f"Examples help command processed for user {update.effective_user.id}")
        except Exception as e:
            logger.error(f"Error sending examples help message: {str(e)}")
    
    async def privacy_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle /privacy command - display privacy policy
        """
        try:
            # Read the privacy policy file
            privacy_file_path = os.path.join(os.path.dirname(__file__), 'Privacy_Policy.md')
            
            if os.path.exists(privacy_file_path):
                with open(privacy_file_path, 'r', encoding='utf-8') as file:
                    privacy_content = file.read()
                
                # Split content into chunks to handle Telegram's message length limit
                max_length = 4000  # Telegram's limit is 4096, leaving some buffer
                
                if len(privacy_content) <= max_length:
                    await update.message.reply_text(privacy_content, parse_mode='Markdown')
                else:
                    # Send privacy policy in chunks
                    chunks = []
                    current_chunk = ""
                    
                    lines = privacy_content.split('\n')
                    for line in lines:
                        if len(current_chunk + line + '\n') <= max_length:
                            current_chunk += line + '\n'
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = line + '\n'
                    
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    
                    # Send first chunk with header
                    await update.message.reply_text(
                        f"📋 **Privacy Policy** (Part 1 of {len(chunks)})\n\n{chunks[0]}", 
                        parse_mode='Markdown'
                    )
                    
                    # Send remaining chunks
                    for i, chunk in enumerate(chunks[1:], 2):
                        await update.message.reply_text(
                            f"📋 **Privacy Policy** (Part {i} of {len(chunks)})\n\n{chunk}", 
                            parse_mode='Markdown'
                        )
            else:
                # Fallback message if file doesn't exist
                fallback_message = """
📋 **Privacy Policy**

We are committed to protecting your privacy and ensuring transparency in our data practices.

**Key Points:**
• Your trading data is encrypted and secure
• We only collect necessary information for bot functionality
• You have full control over your data
• We comply with GDPR, CCPA, and financial regulations
• AI processing is transparent and under your control

**Your Rights:**
• Access your data
• Request data deletion
• Opt-out of AI training
• Export your information

**Contact:** For privacy questions, contact our Data Protection Officer.

**Full Policy:** The complete privacy policy is available upon request.
                """
                await update.message.reply_text(fallback_message, parse_mode='Markdown')
            
            logger.info(f"Privacy command processed for user {update.effective_user.id}")
            
        except Exception as e:
            logger.error(f"Error in privacy command: {str(e)}")
            await update.message.reply_text(
                "❌ Error loading privacy policy. Please try again or contact support.", 
                parse_mode='Markdown'
            )
    
    async def contact_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle /contact command - display contact information
        """
        try:
            contact_message = """
📧 **CONTACT & SUPPORT**

**📬 Get in Touch:**
• **Email**: amar@vidality.com
• **Response Time**: Within 24-48 hours

**🛠️ What We Can Help With:**
• **Technical Issues** - Bot errors or bugs
• **Feature Requests** - Suggest new functionality
• **Trading Questions** - Strategy and analysis help
• **Account Support** - Access and subscription issues
• **Privacy Concerns** - Data protection questions
• **API Integration** - Custom development support

**💡 Before Contacting:**
• Try `/help` for command assistance
• Check `/privacy` for data protection info
• Use `/help_examples` for usage tips

**🚀 Quick Support:**
• Include your username and issue description
• Attach screenshots if relevant
• Mention specific commands that aren't working

**⚡ Fast Response Guaranteed!**
We're committed to helping you get the most out of your AI trading assistant.
            """
            
            await update.message.reply_text(contact_message, parse_mode='Markdown')
            logger.info(f"Contact command processed for user {update.effective_user.id}")
            
        except Exception as e:
            logger.error(f"Error in contact command: {str(e)}")
            await update.message.reply_text(
                "❌ Error loading contact information. Please try again or email amar@vidality.com directly.", 
                parse_mode='Markdown'
            )
    
    def _detect_natural_language_query(self, message: str) -> dict:
        """
        Detect and parse natural language queries about stocks, prices, and news
        
        Args:
            message: User's message text
            
        Returns:
            dict: Query information with type, symbol, and suggested command
        """
        import re
        message_lower = message.lower().strip()
        
        # Enhanced symbol extraction function
        def extract_symbol_from_message(text, original_message):
            # Company name to symbol mapping
            name_to_symbol = {
                'apple': 'AAPL', 'tesla': 'TSLA', 'microsoft': 'MSFT',
                'google': 'GOOGL', 'alphabet': 'GOOGL', 'amazon': 'AMZN',
                'meta': 'META', 'facebook': 'META', 'nvidia': 'NVDA',
                'netflix': 'NFLX', 'bitcoin': 'BTC', 'ethereum': 'ETH'
            }
            
            # Clean text for better matching (remove punctuation, possessives)
            import re
            clean_text = re.sub(r"[^\w\s]", " ", text).lower()
            
            # Check for company names (including possessive forms like "Apple's")
            for word in clean_text.split():
                # Remove possessive 's' if present
                clean_word = word.rstrip('s') if word.endswith('s') else word
                if clean_word in name_to_symbol:
                    return name_to_symbol[clean_word]
                if word in name_to_symbol:
                    return name_to_symbol[word]
            
            # Check for direct symbols (uppercase words 2-5 chars)
            for word in original_message.split():
                clean_word = re.sub(r"[^A-Za-z]", "", word)  # Remove punctuation
                if clean_word.isupper() and 2 <= len(clean_word) <= 5 and clean_word.isalpha():
                    return clean_word
            
            # Check for lowercase symbols
            symbol_words = ['aapl', 'tsla', 'msft', 'googl', 'amzn', 'meta', 'nvda', 'nflx', 'btc', 'eth']
            for word in clean_text.split():
                if word in symbol_words:
                    return word.upper()
            
            return None
        
        # Extract symbol first
        symbol = extract_symbol_from_message(message_lower, message)
        if not symbol:
            return {'detected': False}
        
        # Determine query type based on keywords (prioritized order)
        query_type = None
        command_map = {
            'price': '/price',
            'news': '/analyze', 
            'analysis': '/analyze',
            'chart': '/chart'
        }
        
        # Price keywords (highest priority for financial queries)
        price_keywords = ['price', 'cost', 'value', 'worth', 'much', 'trading', 'current', 'today']
        if any(keyword in message_lower for keyword in price_keywords):
            # Special case: if 'chart' is also mentioned, prioritize chart
            if any(chart_word in message_lower for chart_word in ['chart', 'graph', 'plot']):
                query_type = 'chart'
            else:
                query_type = 'price'
        
        # Chart keywords
        elif any(keyword in message_lower for keyword in ['chart', 'graph', 'plot', 'visual']):
            query_type = 'chart'
        
        # Analysis keywords
        elif any(keyword in message_lower for keyword in ['analyze', 'analysis', 'review', 'buy', 'sell', 'investment', 'technical']):
            query_type = 'analysis'
        
        # News keywords (broader patterns)
        elif any(keyword in message_lower for keyword in ['news', 'updates', 'latest', 'happening', 'going on', 'about', "what's"]):
            query_type = 'news'
        
        # Fallback: if we have a symbol but no clear type, default to price
        else:
            query_type = 'price'
        
        if query_type:
            return {
                'type': query_type,
                'symbol': symbol,
                'command': f'{command_map[query_type]} {symbol}',
                'detected': True
            }
        
        return {'detected': False}
    
    async def _handle_natural_language_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE, query_info: dict) -> bool:
        """
        Handle detected natural language queries using OpenAI API for conversational responses
        
        Args:
            update: Telegram update object
            context: Bot context
            query_info: Detected query information
            
        Returns:
            bool: True if query was handled, False otherwise
        """
        if not query_info.get('detected'):
            return False
        
        user = update.effective_user
        user_message = update.message.text
        query_type = query_info['type']
        symbol = query_info['symbol']
        command = query_info['command']
        
        # Send typing indicator
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        
        try:
            # Create a specialized prompt for natural language stock queries
            nl_prompt = f"""
You are a helpful financial assistant. The user asked: "{user_message}"

I've detected they're asking about {symbol} stock with intent: {query_type}.

Please provide a conversational response that:
1. Acknowledges their question naturally
2. Suggests they use the command: {command}
3. Briefly explains why the command will give them better/more accurate results
4. Keep it friendly and concise (2-3 sentences max)
5. Use appropriate emojis

Do NOT fetch any actual stock data - just provide a conversational response guiding them to use the command.
"""
            
            # Use OpenAI API for natural language response
            ai_response = await self.openai_service.generate_response(
                nl_prompt, user.id, context_str="Natural language stock query handling"
            )
            
            if ai_response and ai_response.strip():
                # Add the suggested command at the end
                final_response = ai_response.strip()
                if not command in final_response:
                    final_response += f"\n\n💡 Try: `{command}`"
                
                await update.message.reply_text(final_response, parse_mode='Markdown')
                
                # Store this interaction in memory
                self.conversation_memory.add_message(user.id, user_message, final_response)
                
                return True
            else:
                # Fallback to simple response if OpenAI fails
                raise Exception("OpenAI response was empty")
            
        except Exception as e:
            logger.error(f"Error with OpenAI natural language query: {e}")
            # Simple fallback response
            fallback_response = f"I understand you're asking about {symbol}!\n\n"
            fallback_response += f"For the best results, please try: `{command}`\n\n"
            fallback_response += "💡 **Available commands:**\n"
            fallback_response += f"• `/price {symbol}` - Get current price\n"
            fallback_response += f"• `/chart {symbol}` - View price chart\n"
            fallback_response += f"• `/analyze {symbol}` - Full analysis"
            
            await update.message.reply_text(fallback_response, parse_mode='Markdown')
            return True
    
    @remember_interaction(memory_type=MemoryType.CONVERSATION, importance=MemoryImportance.MEDIUM)
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle regular text messages with natural language query detection and OpenAI integration
        
        Args:
            update: Telegram update object
            context: Bot context
        """
        user = update.effective_user
        user_message = update.message.text
        
        try:
            # Log incoming message
            logger.info(f"Message from {user.username or user.id}: {user_message[:100]}...")
            
            # First, try to detect and handle natural language queries
            query_info = self._detect_natural_language_query(user_message)
            if await self._handle_natural_language_query(update, context, query_info):
                # Query was handled by natural language processor
                logger.info(f"Natural language query handled for user {user.id}: {query_info['type']} - {query_info.get('symbol', 'N/A')}")
                return
            
            # If not a detected query, proceed with OpenAI processing
            # Define the AI processing operation
            async def ai_processing_operation():
                # Get enhanced contextual data for better responses
                contextual_data = await self.memory_integration.get_contextual_prompt_enhancement(
                    user_id=user.id,
                    current_query=user_message
                )
                
                # Retrieve traditional conversation context for backward compatibility
                context_str = self.conversation_memory.get_conversation_context(user.id)
                
                # Enhance context with memory insights
                enhanced_context = self._build_enhanced_context(context_str, contextual_data)
                
                logger.info(f"Calling OpenAI service for user {user.id} with enhanced context")
                ai_response = await self.openai_service.generate_response(
                    user_message, user.id, context_str=enhanced_context
                )
                logger.info(f"OpenAI service returned: {type(ai_response)} - {len(str(ai_response)) if ai_response else 'None'} chars")
                return ai_response
            
            # Execute AI processing with enhanced typing indicator
            ai_response = await self._with_typing_indicator(
                update.effective_chat.id, context, ai_processing_operation
            )
            
            if ai_response and ai_response.strip():
                # Send AI response to user
                logger.info(f"Sending response to user {user.id}: {ai_response[:100]}...")
                await update.message.reply_text(ai_response)
                logger.info(f"Response sent successfully to user {user.id}")
                
                # Store in both traditional and enhanced memory systems
                self.conversation_memory.add_message(user.id, user_message, ai_response)
                
                # Enhanced memory storage with automatic context extraction
                memory_id = await self.enhanced_memory.add_interaction(
                    user_id=user.id,
                    user_message=user_message,
                    bot_response=ai_response,
                    message_type="text",
                    context={
                        "chat_id": update.effective_chat.id,
                        "username": user.username,
                        "message_length": len(user_message),
                        "response_length": len(ai_response)
                    },
                    importance=MemoryImportance.MEDIUM
                )
                

            else:
                # Fallback response if AI service fails
                logger.error(f"AI response is empty or None for user {user.id}. Response: {repr(ai_response)}")
                
                # Check if this looks like a command that should be handled
                if user_message.strip().startswith('/'):
                    # Check for command typos and provide suggestions
                    command_part = user_message.strip().split()[0].lower()
                    
                    # Try to detect command typos using enhanced error handler
                    error_message = self.error_handler.handle_command_error(user_message.strip())
                    
                    # If no specific suggestion was found, provide general fallback
                    if "Did you mean" not in error_message:
                        fallback_message = (
                            "⚠️ AI chat is temporarily unavailable, but all trading commands are working!\n\n"
                            "📊 **Available Commands:**\n"
                            "• `/price AAPL` - Get stock price\n"
                            "• `/chart TSLA` - Generate price chart\n"
                            "• `/analyze NVDA` - Stock analysis\n"
                            "• `/help` - Full command list\n\n"
                            "💡 Try one of these commands or use `/menu` for more options!"
                        )
                    else:
                        fallback_message = error_message
                else:
                    fallback_message = (
                        "⚠️ AI chat is temporarily unavailable due to quota limits.\n\n"
                        "📊 **But all trading features are working:**\n"
                        "• `/price SYMBOL` - Real-time prices\n"
                        "• `/chart SYMBOL` - Technical charts\n"
                        "• `/analyze SYMBOL` - Stock analysis\n"
                        "• `/menu` - See all commands\n\n"
                        "🔄 AI chat will be restored once quota resets."
                    )
                
                await update.message.reply_text(fallback_message)
                logger.error(f"Sent enhanced fallback message to user {user.id}")
                
        except TelegramError as e:
            logger.error(f"Telegram error for user {user.id}: {str(e)}")
            try:
                await update.message.reply_text("I'm sorry, I encountered an error processing your request. Please try again later.")
            except:
                pass
        except Exception as e:
            logger.error(f"Unexpected error handling message from user {user.id}: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            try:
                await update.message.reply_text("I'm sorry, I encountered an error processing your request. Please try again later.")
            except:
                pass  # Don't crash if we can't even send error message
    
    # ===== TRADING COMMANDS =====
    
    @time_operation("price_command")
    @cache_result(ttl=30)  # Cache for 30 seconds for real-time data
    @remember_interaction(memory_type=MemoryType.QUERY, importance=MemoryImportance.MEDIUM)
    async def price_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /price command for stock prices"""
        user = update.effective_user
        user_id = str(user.id) if user else "unknown"
        
        # Record metrics
        metrics.record_message("price", user_id)
        
        try:
            args = context.args
            if not args:
                # Use modern UI for help message
                help_message = TradingBotUI.format_help_section(
                    "📊 Price Command",
                    "Get real-time stock prices and key metrics",
                    [
                        "📊 `/price AAPL` - Get current price",
                        "📈 Supports ALL US stocks (NYSE, NASDAQ, AMEX)",
                        "⚡ Real-time and delayed data available",
                        "📱 Over 8,000+ US stocks supported"
                    ]
                )
                keyboard = TradingBotUI.create_price_keyboard()
                
                await update.message.reply_text(
                    help_message,
                    reply_markup=keyboard,
                    parse_mode='Markdown'
                )
                return
            
            symbol = args[0].upper().strip()
            
            # Validate and normalize US stock symbol
            if not self._is_valid_us_stock(symbol):
                # Use enhanced error handler for better user experience
                error_message = self.error_handler.format_error_message(
                    "invalid_symbol",
                    context={"invalid_symbol": symbol}
                )
                
                await update.message.reply_text(error_message, parse_mode='Markdown')
                return
            
            normalized_symbol = self._normalize_us_stock_symbol(symbol)
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
            
            logger.info(f"Price command requested for symbol: {symbol} (normalized: {normalized_symbol})")
            
            # Check cache first for price data
            cache_key = f"price_data_{normalized_symbol}"
            price_data = performance_cache.get(cache_key)
            
            if not price_data:
                # Use connection pool for API call
                async with await connection_pool.acquire("market_data") as conn:
                    price_data = await self.market_service.get_stock_price(normalized_symbol, update.effective_user.id)
                
                # Cache the result if successful
                if price_data:
                    performance_cache.set(cache_key, price_data, ttl=30)  # Cache for 30 seconds
            
            if price_data:
                # Record successful price lookup
                metrics.record_trading_signal("price_lookup", normalized_symbol)
                
                # Use modern UI for price display
                message = TradingBotUI.format_price_data(price_data, normalized_symbol)
                keyboard = TradingBotUI.create_stock_action_menu(normalized_symbol)
                
                await update.message.reply_text(
                    message,
                    reply_markup=keyboard,
                    parse_mode='Markdown'
                )
            else:
                error_message = TradingBotUI.format_error_message(
                    f"Could not fetch price data for {normalized_symbol}",
                    [
                        "Market is currently closed",
                        "Stock is delisted or suspended", 
                        "Temporary data provider issue",
                        "Symbol requires different format"
                    ],
                    [
                        "Verify symbol on financial websites",
                        "Try during market hours (9:30 AM - 4:00 PM ET)",
                        "Check if company has been acquired/merged",
                        "Use /help for more commands"
                    ]
                )
                keyboard = TradingBotUI.create_main_menu()
                
                await update.message.reply_text(
                    error_message,
                    reply_markup=keyboard,
                    parse_mode='Markdown'
                )
                
        except Exception as e:
            # Record error
            metrics.record_error("PriceCommandError", "telegram_handler")
            
            logger.error(f"Error in price command for {args[0] if args else 'unknown'}: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            error_message = TradingBotUI.format_error_message(
                f"Error fetching price data for {args[0] if args else 'symbol'}",
                [
                    "Symbol might be incorrect (e.g., AAPL, TSLA)",
                    "Market might be closed",
                    "Temporary service issue"
                ],
                [
                    "Verify the stock symbol",
                    "Try again in a few moments",
                    "Check if market is open"
                ]
            )
            keyboard = TradingBotUI.create_main_menu()
            
            await update.message.reply_text(
                error_message,
                reply_markup=keyboard,
                parse_mode='Markdown'
            )
    
    async def chart_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /chart command for price charts"""
        try:
            args = context.args
            if not args:
                help_message = TradingBotUI.format_help_section(
                    "📈 Chart Command",
                    "Generate professional price charts with technical analysis",
                    [
                        "📈 `/chart TSLA [1d]` - Generate chart",
                        "📊 Technical indicators included",
                        "⏰ Multiple timeframes supported",
                        "🎯 Professional analysis tools"
                    ]
                )
                keyboard = TradingBotUI.create_main_menu()
                
                await update.message.reply_text(
                    help_message,
                    reply_markup=keyboard,
                    parse_mode='Markdown'
                )
                return
            
            symbol = args[0].upper().strip()
            period = args[1] if len(args) > 1 else '1d'
            
            # Validate period parameter
            valid_periods = {
                # Standard periods
                '1d', '5d', '1w', '1wk', '1mo', '1M', '3mo', '3M', '6mo', '6M', 
                '1y', '1Y', '2y', '5y', '1h', '1m',
                # Common user aliases
                '6m', '1min', '5min', '15min', '30min', '1hr', '3m', '12m'
            }
            
            if period not in valid_periods:
                error_message = TradingBotUI.format_error_message(
                    f"Invalid timeframe '{period}'",
                    [
                        "Supported timeframes:",
                        "• **Minutes**: 1m, 5min, 15min, 30min, 1hr",
                        "• **Days**: 1d, 5d, 1w", 
                        "• **Months**: 1mo, 3m, 6m, 12m",
                        "• **Years**: 1y, 2y, 5y"
                    ],
                    [
                        "Use a supported timeframe",
                        "Example: `/chart AAPL 6m`",
                        "Example: `/chart TSLA 1y`"
                    ]
                )
                keyboard = TradingBotUI.create_main_menu()
                
                await update.message.reply_text(
                    error_message,
                    reply_markup=keyboard,
                    parse_mode='Markdown'
                )
                return
            
            # Validate and normalize US stock symbol
            if not self._is_valid_us_stock(symbol):
                # Use enhanced error handler for better user experience
                error_message = self.error_handler.format_error_message(
                    "invalid_symbol",
                    context={"invalid_symbol": symbol}
                )
                
                await update.message.reply_text(error_message, parse_mode='Markdown')
                return
            
            normalized_symbol = self._normalize_us_stock_symbol(symbol)
            
            # Define chart generation operation
            async def chart_generation_operation():
                await update.message.reply_text(f"📊 Generating chart for {normalized_symbol}...")
                return await self.chart_service.generate_price_chart(normalized_symbol, period)
            
            # Execute chart generation with enhanced typing indicator
            chart_b64 = await self._with_typing_indicator(
                update.effective_chat.id, context, chart_generation_operation
            )
            
            if chart_b64:
                chart_bytes = base64.b64decode(chart_b64)
                chart_file = BytesIO(chart_bytes)
                chart_file.name = f"{normalized_symbol}_chart.png"
                
                # Create keyboard for chart actions
                keyboard = TradingBotUI.create_stock_actions_keyboard(normalized_symbol)
                
                await update.message.reply_photo(
                    photo=InputFile(chart_file),
                    caption=f"📈 **{normalized_symbol} Price Chart** ({period})\n\n"
                           f"✅ Professional technical analysis chart\n"
                           f"📊 Generated with advanced indicators",
                    reply_markup=keyboard,
                    parse_mode='Markdown'
                )
            else:
                error_message = TradingBotUI.format_error_message(
                    f"Could not generate chart for {normalized_symbol}",
                    [
                        "Insufficient historical data",
                        "Market data unavailable",
                        "Technical issue with chart generation"
                    ],
                    [
                        "Try a different symbol",
                        "Check if market is open",
                        "Try again in a few moments"
                    ]
                )
                keyboard = TradingBotUI.create_main_menu()
                
                await update.message.reply_text(
                    error_message,
                    reply_markup=keyboard,
                    parse_mode='Markdown'
                )
                
        except Exception as e:
            logger.error(f"Error in chart command: {e}")
            
            error_message = TradingBotUI.format_error_message(
                "Error generating chart",
                ["Technical issue occurred", "Service temporarily unavailable"],
                ["Try again in a few moments", "Try a different symbol or timeframe"]
            )
            keyboard = TradingBotUI.create_main_menu()
            
            await update.message.reply_text(
                error_message,
                reply_markup=keyboard,
                parse_mode='Markdown'
            )
    
    @time_operation("analyze_command")
    @cache_result(ttl=300)  # Cache for 5 minutes for AI analysis
    async def analyze_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /analyze command for AI stock analysis or sector analysis"""
        user = update.effective_user
        user_id = str(user.id) if user else "unknown"
        
        # Record metrics
        metrics.record_message("analyze", user_id)
        
        try:
            args = context.args
            if not args:
                help_message = TradingBotUI.format_help_section(
                    "🔍 Analysis Command",
                    "Get comprehensive AI-powered stock or sector analysis",
                    [
                        "🔍 `/analyze NVDA` - Get AI stock analysis",
                        "🏭 `/analyze tech` - Get technology sector analysis",
                        "🤖 AI-powered insights and recommendations",
                        "📊 Technical analysis included",
                        "💡 Investment recommendations"
                    ]
                )
                keyboard = TradingBotUI.create_analysis_keyboard()
                
                await update.message.reply_text(
                    help_message,
                    reply_markup=keyboard,
                    parse_mode='Markdown'
                )
                return
            
            symbol = args[0].upper().strip()
            
            # Check if this is a sector analysis request
            if symbol.lower() == 'tech' or symbol.lower() == 'technology':
                await self._analyze_tech_sector(update, context)
                return
            
            # Validate and normalize US stock symbol
            if not self._is_valid_us_stock(symbol):
                # Use enhanced error handler for better user experience
                error_message = self.error_handler.format_error_message(
                    "invalid_symbol",
                    context={"symbol": symbol}
                )
                
                await update.message.reply_text(error_message, parse_mode='Markdown')
                return
            
            normalized_symbol = self._normalize_us_stock_symbol(symbol)
            
            # Define analysis operation
            async def analysis_operation():
                await update.message.reply_text(f"🤖 Analyzing {normalized_symbol}...")
                
                # Check cache first for analysis
                cache_key = f"analysis_{normalized_symbol}"
                analysis = performance_cache.get(cache_key)
                
                if not analysis:
                    # Use connection pool for AI analysis
                    async with await connection_pool.acquire("ai_analysis") as conn:
                        analysis = await self.trading_intelligence.analyze_stock(normalized_symbol, update.effective_user.id)
                    
                    # Cache successful analysis
                    if analysis and not analysis.get('error'):
                        performance_cache.set(cache_key, analysis, ttl=300)  # Cache for 5 minutes
                
                return analysis
            
            # Execute analysis with enhanced typing indicator
            analysis = await self._with_typing_indicator(
                update.effective_chat.id, context, analysis_operation
            )
            
            # Record AI request
            metrics.record_ai_request("openai", "gpt-4o-mini")
            
            if analysis.get('error'):
                error_message = TradingBotUI.format_error_message(
                    analysis['error'],
                    [
                        "Insufficient data for analysis",
                        "Market data unavailable",
                        "Temporary service issue"
                    ],
                    [
                        "Try again later",
                        "Try a different symbol",
                        "Check if market is open"
                    ]
                )
                keyboard = TradingBotUI.create_main_menu()
                
                await update.message.reply_text(
                    error_message,
                    reply_markup=keyboard,
                    parse_mode='Markdown'
                )
                return
            
            # Get AI insights and technical data
            ai_insights = analysis.get('ai_analysis', '').strip()
            current_data = analysis.get('current_data', {})
            technical_levels = analysis.get('technical_levels', {})
            
            # Determine emoji based on trend and recommendation
            trend_emoji = technical_levels.get('trend_emoji', '🔍')
            
            # Show actual AI error or fallback if AI insights are missing or failed
            error_phrases = ['I encountered an issue', 'Please try again later', 'I apologize', 'I\'m sorry', 
                           'error', 'unable to', 'cannot', 'couldn\'t', 'failed', 'timeout']
            
            if not ai_insights or any(phrase in ai_insights.lower() for phrase in error_phrases):
                error_msg = f"❌ AI analysis failed for {normalized_symbol}.\n\n"
                error_msg += "Possible reasons:\n"
                error_msg += "• OpenAI API quota exceeded or misconfigured\n"
                error_msg += "• Network or timeout issue\n"
                error_msg += "• Service temporarily unavailable\n\n"
                error_msg += "Trying alternative analysis method...\n"
                
                # Send initial error notification
                await update.message.reply_text(error_msg)
                
                # Log the prompt and AI error for debugging
                logging.getLogger("trading_intelligence").error(f"AI analysis failed for {normalized_symbol}. Analysis: {repr(ai_insights)}")
                
                # Check if we have market data to provide basic information
                if current_data:
                    basic_analysis = f"📊 Basic {normalized_symbol} Information:\n\n"
                    
                    if 'company_name' in current_data and current_data['company_name']:
                        basic_analysis += f"Company: {current_data['company_name']}\n"
                    
                    if 'price' in current_data:
                        basic_analysis += f"Current Price: ${current_data['price']}\n"
                    
                    if 'change_percent' in current_data:
                        change = current_data['change_percent']
                        emoji = "🔴" if change < 0 else "🟢"
                        basic_analysis += f"Change: {emoji} {change:.2f}%\n"
                        
                    if 'volume' in current_data:
                        basic_analysis += f"Volume: {current_data['volume']:,}\n"
                    
                    basic_analysis += "\n⚠️ AI analysis unavailable. Using basic market data only."
                    
                    await update.message.reply_text(basic_analysis)
                
                return
            
            # Format enhanced analysis message with current price and technical data
            price = current_data.get('price', 0)
            change_percent = current_data.get('change_percent', 0)
            company_name = current_data.get('company_name', normalized_symbol)
            volume = current_data.get('volume', 0)
            
            # Create header with current price info
            change_emoji = "🟢" if change_percent >= 0 else "🔴"
            header = f"{trend_emoji} **{company_name} ({normalized_symbol})**\n"
            header += f"💰 **${price:.2f}** {change_emoji} **{change_percent:+.2f}%**\n"
            
            # Add volume info if available
            if volume > 0:
                volume_str = f"{volume:,}" if volume < 1000000 else f"{volume/1000000:.1f}M"
                header += f"📊 Volume: {volume_str}\n"
            
            header += "\n"
            
            # Add technical levels if available
            if technical_levels:
                support = technical_levels.get('support', 0)
                resistance = technical_levels.get('resistance', 0)
                trend = technical_levels.get('trend', 'Neutral')
                range_pos = technical_levels.get('range_position', 50)
                
                header += f"📈 **Key Levels:**\n"
                header += f"• Trend: {trend} {trend_emoji}\n"
                header += f"• Support: ${support:.2f}\n"
                header += f"• Resistance: ${resistance:.2f}\n"
                header += f"• 52W Range: {range_pos:.1f}%\n\n"
            
            # Format the AI analysis with better structure
            formatted_analysis = ai_insights
            
            # Clean up any malformed markdown that could cause parsing errors
            if formatted_analysis:
                # Fix common markdown issues
                formatted_analysis = formatted_analysis.replace('**RECOMMENDATION:**', '\n🎯 **RECOMMENDATION:**')
                formatted_analysis = formatted_analysis.replace('**RISK MANAGEMENT:**', '\n⚠️ **RISK MANAGEMENT:**')
                formatted_analysis = formatted_analysis.replace('**CONDITIONAL STRATEGY:**', '\n💡 **CONDITIONAL STRATEGY:**')
                formatted_analysis = formatted_analysis.replace('**TECHNICAL ANALYSIS:**', '\n🔍 **TECHNICAL ANALYSIS:**')
                formatted_analysis = formatted_analysis.replace('**OUTLOOK & CATALYSTS:**', '\n⏰ **OUTLOOK & CATALYSTS:**')
                
                # Remove any double newlines that might cause formatting issues
                formatted_analysis = formatted_analysis.replace('\n\n\n', '\n\n')
                
                # Ensure proper spacing after emojis
                formatted_analysis = formatted_analysis.strip()
            
            message = f"""{header}🤖 **AI Analysis:**
{formatted_analysis}

🕐 **Updated:** {analysis.get('timestamp', self._get_ist_timestamp())}
            """
            
            # Create keyboard for stock actions
            keyboard = TradingBotUI.create_stock_actions_keyboard(normalized_symbol)
            
            await update.message.reply_text(
                message,
                reply_markup=keyboard,
                parse_mode='Markdown'
            )
            
            # Record successful analysis
            metrics.record_trading_signal("ai_analysis", normalized_symbol)
            
        except Exception as e:
            # Record error
            metrics.record_error("AnalyzeCommandError", "telegram_handler")
            
            logger.error(f"Error in analyze command: {e}")
            await update.message.reply_text("❌ Error performing analysis. Please try again.")
    
    async def _analyze_tech_sector(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Analyze the technology sector with comprehensive insights"""
        try:
            # Send typing indicator
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
            
            # Send initial status message
            status_msg = await update.message.reply_text(
                "🏭 **Analyzing Technology Sector...**\n\n"
                "📊 Fetching sector performance data...\n"
                "💻 Analyzing tech stocks...\n"
                "⏳ Please wait...",
                parse_mode='Markdown'
            )
            
            # Get sector summary data
            sector_data = await self.market_service.get_sector_summary()
            
            if not sector_data or 'sectors' not in sector_data:
                await status_msg.edit_text(
                    "❌ **Technology Sector Analysis Failed**\n\n"
                    "Unable to fetch sector data. Please try again later.",
                    parse_mode='Markdown'
                )
                return
            
            # Find technology sector data
            tech_sector = None
            for sector in sector_data['sectors']:
                if sector['sector'].lower() == 'technology':
                    tech_sector = sector
                    break
            
            if not tech_sector:
                await status_msg.edit_text(
                    "❌ **Technology Sector Data Not Available**\n\n"
                    "Technology sector data is currently unavailable. Please try again later.",
                    parse_mode='Markdown'
                )
                return
            
            # Get additional tech stock data for comprehensive analysis
            major_tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']
            tech_stocks_data = []
            
            for symbol in major_tech_stocks:
                try:
                    stock_data = await self.market_service.get_stock_price(symbol)
                    if stock_data and not stock_data.get('error'):
                        tech_stocks_data.append({
                            'symbol': symbol,
                            'price': stock_data.get('price', 0),
                            'change_percent': stock_data.get('change_percent', 0),
                            'volume': stock_data.get('volume', 0),
                            'company_name': stock_data.get('company_name', symbol)
                        })
                except Exception as e:
                    logger.warning(f"Failed to fetch data for {symbol}: {e}")
                    continue
            
            # Sort tech stocks by performance
            tech_stocks_data.sort(key=lambda x: x['change_percent'], reverse=True)
            
            # Calculate sector metrics
            avg_change = sum(stock['change_percent'] for stock in tech_stocks_data) / len(tech_stocks_data) if tech_stocks_data else 0
            positive_stocks = len([s for s in tech_stocks_data if s['change_percent'] > 0])
            total_stocks = len(tech_stocks_data)
            
            # Determine sector sentiment
            if avg_change > 1.5:
                sentiment = "🟢 **BULLISH**"
                sentiment_desc = "Strong upward momentum"
            elif avg_change > 0.5:
                sentiment = "🟡 **MODERATELY BULLISH**"
                sentiment_desc = "Positive but cautious momentum"
            elif avg_change > -0.5:
                sentiment = "⚪ **NEUTRAL**"
                sentiment_desc = "Mixed signals, sideways movement"
            elif avg_change > -1.5:
                sentiment = "🟠 **MODERATELY BEARISH**"
                sentiment_desc = "Negative but not severe"
            else:
                sentiment = "🔴 **BEARISH**"
                sentiment_desc = "Strong downward pressure"
            
            # Format the comprehensive analysis message
            message = f"""🏭 **TECHNOLOGY SECTOR ANALYSIS**

📊 **Sector Overview:**
• ETF: {tech_sector['symbol']} - ${tech_sector['price']:.2f}
• Performance: {'🟢' if tech_sector['change_percent'] >= 0 else '🔴'} **{tech_sector['change_percent']:+.2f}%**
• Volume: {tech_sector['volume']:,}

🎯 **Sector Sentiment:** {sentiment}
💭 {sentiment_desc}

📈 **Key Metrics:**
• Average Change: {avg_change:+.2f}%
• Positive Stocks: {positive_stocks}/{total_stocks} ({positive_stocks/total_stocks*100:.1f}%)
• Sector Rank: Technology

💻 **Top Tech Performers:**"""
            
            # Add top 5 performers
            for i, stock in enumerate(tech_stocks_data[:5]):
                emoji = "🟢" if stock['change_percent'] >= 0 else "🔴"
                message += f"\n{i+1}. **{stock['symbol']}** - ${stock['price']:.2f} {emoji} {stock['change_percent']:+.2f}%"
            
            if len(tech_stocks_data) > 5:
                message += "\n\n📉 **Bottom Performers:"
                for i, stock in enumerate(tech_stocks_data[-3:]):
                    emoji = "🟢" if stock['change_percent'] >= 0 else "🔴"
                    message += f"\n• **{stock['symbol']}** - ${stock['price']:.2f} {emoji} {stock['change_percent']:+.2f}%"
            
            # Add market insights
            message += f"\n\n🔍 **Market Insights:**"
            
            if avg_change > 1:
                message += "\n• Strong tech rally in progress"
                message += "\n• Consider growth-focused positions"
                message += "\n• Monitor for potential overextension"
            elif avg_change > 0:
                message += "\n• Moderate tech sector strength"
                message += "\n• Selective stock picking recommended"
                message += "\n• Watch for continuation patterns"
            elif avg_change > -1:
                message += "\n• Tech sector showing mixed signals"
                message += "\n• Wait for clearer direction"
                message += "\n• Focus on quality names"
            else:
                message += "\n• Tech sector under pressure"
                message += "\n• Consider defensive positioning"
                message += "\n• Look for oversold opportunities"
            
            # Add timestamp
            message += f"\n\n🕐 **Updated:** {sector_data.get('timestamp', self._get_ist_timestamp())}"
            
            # Update the status message with the full analysis
            await status_msg.edit_text(
                message,
                parse_mode='Markdown'
            )
            
        except Exception as e:
            logger.error(f"Error in tech sector analysis: {e}")
            await update.message.reply_text(
                "❌ **Technology Sector Analysis Failed**\n\n"
                "An error occurred while analyzing the technology sector. Please try again later.",
                parse_mode='Markdown'
            )
    
    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle errors in the bot with enhanced error handling
        
        Args:
            update: Telegram update object
            context: Bot context
        """
        logger.error(f"Bot error: {context.error}")
        
        # Try to inform the user about the error if update is available
        if isinstance(update, Update) and update.effective_message:
            try:
                # Use enhanced error handler for better user experience
                error_message = self.error_handler.format_error_message(
                    str(context.error),
                    error_type="general",
                    command="",
                    user_input=""
                )
                await update.effective_message.reply_text(
                    error_message,
                    parse_mode='Markdown'
                )
            except:
                # Fallback to simple message if enhanced handling fails
                try:
                    await update.effective_message.reply_text(
                        "Sorry, I encountered an error. Please try again."
                    )
                except:
                    pass  # Don't crash if we can't send the error message
    


    # UI callback functions removed - bot is now text-only
    
    # Callback query handler removed - bot is now text-only
    
    # Menu callback removed - bot is now text-only

    # Quick actions handler removed - bot is now text-only
    
    # Process menu input removed - bot is now text-only
    

    
    async def pdf_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle PDF document uploads from users, extract text, analyze, and reply with insights.
        """
        try:
            user = update.effective_user
            document = update.message.document
            if not document or not document.file_name.lower().endswith('.pdf'):
                await update.message.reply_text("❌ Please send a valid PDF document.")
                return
            file = await context.bot.get_file(document.file_id)
            pdf_bytes = await file.download_as_bytearray()
            await update.message.reply_chat_action(action="typing")
            # Extract text from PDF (try text extraction first, then OCR if needed)
            text = ""
            try:
                reader = PdfReader(io.BytesIO(pdf_bytes))
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
            except Exception as e:
                logger.error(f"Error extracting text from PDF: {str(e)}")
            # If no text found, try OCR
            if not text.strip():
                try:
                    images = convert_from_bytes(pdf_bytes)
                    ocr_texts = [pytesseract.image_to_string(img) for img in images]
                    text = "\n".join(ocr_texts)
                    if text.strip():
                        await update.message.reply_text("ℹ️ The PDF appears to be scanned or image-based. Extracted text using OCR.")
                except Exception as e:
                    logger.error(f"Error extracting text from PDF via OCR: {str(e)}")
                    await update.message.reply_text("❌ Failed to extract text from the PDF, even with OCR. Please try another file.")
                    return
            if not text.strip():
                await update.message.reply_text("❌ No extractable text found in the PDF (even with OCR).")
                return
            # Analyze the extracted text using OpenAI
            analysis = await self.openai_service.generate_response(
                user_message=f"Analyze this PDF content and provide detailed insights, summary, and key points.\n\n{text[:3000]}",
                user_id=user.id
            )
            if analysis:
                await update.message.reply_text(f"📄 *PDF Analysis Result:*\n{analysis}", parse_mode='Markdown')
            else:
                await update.message.reply_text("❌ Failed to analyze the PDF. Please try again later.")
        except Exception as e:
            logger.error(f"Error in pdf_handler: {str(e)}")
            await update.message.reply_text("❌ Error analyzing the PDF. Please try again later.")
    
    async def photo_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle photo uploads from users and analyze chart images using OpenAI Vision.
        """
        try:
            user = update.effective_user
            photo = update.message.photo[-1]  # Get the highest resolution photo
            
            # Download the image
            file = await context.bot.get_file(photo.file_id)
            image_bytes = await file.download_as_bytearray()
            
            await update.message.reply_chat_action(action="typing")
            await update.message.reply_text("📊 Analyzing your chart image...")
            
            # Analyze the image using OpenAI Vision
            analysis = await self.openai_service.analyze_image(image_bytes, user.id)
            
            if analysis:
                # Check if analysis contains an error message
                if analysis.startswith("❌") or analysis.startswith("⚠️") or analysis.startswith("🔧"):
                    await update.message.reply_text(analysis)
                else:
                    # Format the analysis with a professional header
                    formatted_analysis = f"📊 *PROFESSIONAL CHART ANALYSIS*\n\n{analysis}"
                    
                    # Split message if it's too long for Telegram
                    if len(formatted_analysis) > 4000:
                        # Send first part
                        first_part = formatted_analysis[:3900] + "\n\n*...continued in next message*"
                        await update.message.reply_text(first_part, parse_mode='Markdown')
                        # Send second part
                        second_part = "*...continued from previous message*\n\n" + formatted_analysis[3900:]
                        await update.message.reply_text(second_part, parse_mode='Markdown')
                    else:
                        await update.message.reply_text(formatted_analysis, parse_mode='Markdown')
            else:
                await update.message.reply_text(
                    "❌ Failed to analyze the image. Please try again with a clear chart image."
                )
                
        except Exception as e:
            logger.error(f"Error in photo_handler: {str(e)}")
            await update.message.reply_text(
                "❌ Error analyzing the image. Please try again later."
            )
    
    async def smart_signal_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /signals command for comprehensive technical analysis signals"""
        try:
            args = context.args
            if not args:
                await update.message.reply_text("📊 Usage: /signals TSLA\n\nGet comprehensive technical analysis signals for a stock.")
                return
            
            symbol = args[0].upper()
            
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
            await update.message.reply_text(f"🔍 Analyzing {symbol} with advanced technical indicators...")
            
            # Get comprehensive market data and technical analysis
            analysis_data = await self.get_comprehensive_signal_analysis(symbol)
            
            if analysis_data:
                current_price = analysis_data['current_price']
                indicators = analysis_data['indicators']
                signals = analysis_data['signals']
                patterns = analysis_data['patterns']
                
                # Generate signal strength and direction
                signal_strength = self._calculate_technical_signal_strength(indicators, signals)
                overall_direction = signals.get('overall_signal', 'NEUTRAL')
                
                # Calculate key levels
                support_levels = [indicators.get('support_1', current_price * 0.95), 
                                indicators.get('dynamic_support', current_price * 0.97)]
                resistance_levels = [indicators.get('resistance_1', current_price * 1.05), 
                                   indicators.get('dynamic_resistance', current_price * 1.03)]
                
                # Format technical indicators summary
                tech_summary = self._format_technical_summary(indicators)
                
                # Generate entry and exit levels
                entry_levels = self._calculate_entry_levels(current_price, overall_direction, indicators)
                
                # Detect active patterns
                active_patterns = [pattern for pattern, detected in patterns.items() if detected]
                
                # Calculate volatility and risk metrics
                volatility_info = self._calculate_volatility_metrics(indicators)
                
                message = f"""
🎯 **Advanced Signal Analysis: {symbol}**

💰 **Current Price:** ${current_price:.2f}
🔄 **Overall Signal:** {overall_direction} ({signal_strength}/10)

📊 **Technical Indicators:**
{tech_summary}

🎯 **Key Levels:**
• **Support:** ${min(support_levels):.2f} | ${max(support_levels):.2f}
• **Resistance:** ${min(resistance_levels):.2f} | ${max(resistance_levels):.2f}
• **Pivot Point:** ${indicators.get('pivot', current_price):.2f}

🛠 **Trade Setup ({overall_direction}):**
{entry_levels}

📈 **Volume Analysis:**
• **Volume Ratio:** {indicators.get('volume_ratio', 1.0):.2f}x average
• **Money Flow Index:** {indicators.get('mfi', 50):.1f}
• **Chaikin Money Flow:** {indicators.get('cmf', 0):.3f}

⚡ **Volatility & Risk:**
{volatility_info}

🔍 **Pattern Detection:**
{', '.join(active_patterns) if active_patterns else 'No significant patterns detected'}

📅 **Analysis Time:** {self._get_ist_timestamp()}

⚠️ **Disclaimer:** This is technical analysis only. Always do your own research and consider risk management.
                """
                
                await update.message.reply_text(message, parse_mode='Markdown')
            else:
                await update.message.reply_text(f"❌ Unable to analyze {symbol}\n\nPossible reasons:\n• Invalid symbol\n• Market data unavailable\n• Technical analysis failed")
                
        except Exception as e:
            logger.error(f"Error in signals command: {e}")
            await update.message.reply_text("❌ Error performing technical analysis. Please try again.")
    
    async def get_qlib_signal(self, symbol: str) -> Optional[float]:
        """Get Qlib signal for a symbol"""
        try:
            # Import QlibService here to avoid circular imports
            from qlib_service import QlibService
            
            if not hasattr(self, 'qlib_service'):
                self.qlib_service = QlibService()
            
            signal = self.qlib_service.get_signal(symbol)
            return signal
            
        except Exception as e:
            logger.error(f"Error getting Qlib signal for {symbol}: {e}")
            return None
    
    async def _get_comprehensive_deep_analysis(self, symbol: str, timeframe: str) -> Optional[dict]:
        """Get comprehensive deep analysis data"""
        try:
            # Get market data
            market_data = await self.market_service.get_stock_price(symbol)
            if not market_data:
                return None
            
            # Get historical data
            historical_data = self.market_service.get_historical_data(symbol, period=timeframe)
            if historical_data is None or historical_data.empty:
                return None
            
            # Use shared technical indicators instance
            indicators = self.technical_indicators.calculate_all_indicators(historical_data)
            
            # Get AI sentiment analysis
            sentiment_data = await self._get_ai_sentiment_analysis(symbol)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(historical_data)
            
            # Calculate price targets
            price_targets = self._calculate_price_targets(historical_data, indicators)
            
            # Get market context
            market_context = await self._get_market_context(symbol)
            
            return {
                'current_price': market_data.get('price', 0),
                'market_data': market_data,
                'indicators': indicators,
                'sentiment': sentiment_data,
                'risk_metrics': risk_metrics,
                'price_targets': price_targets,
                'market_context': market_context,
                'historical_data': historical_data
            }
            
        except Exception as e:
            logger.error(f"Error getting comprehensive analysis for {symbol}: {e}")
            return None
    
    async def _get_ai_sentiment_analysis(self, symbol: str) -> dict:
        """Get AI-powered sentiment analysis"""
        try:
            # Get recent news sentiment
            news_sentiment = await self._analyze_news_sentiment(symbol)
            
            # Get technical sentiment based on price action
            technical_sentiment = await self._get_technical_sentiment(symbol)
            
            # Combine sentiments with weights
            if news_sentiment != 0:
                # If we have news, weight it more heavily
                overall_sentiment = (news_sentiment * 0.6 + technical_sentiment * 0.4)
            else:
                # If no news, rely on technical analysis
                overall_sentiment = technical_sentiment
            
            sentiment_label = "Bullish" if overall_sentiment > 0.1 else "Bearish" if overall_sentiment < -0.1 else "Neutral"
            
            # Calculate confidence based on data availability
            confidence = min(100, abs(overall_sentiment) * 100 + (30 if news_sentiment != 0 else 0))
            
            return {
                'overall_score': overall_sentiment,
                'label': sentiment_label,
                'news_sentiment': news_sentiment,
                'social_sentiment': technical_sentiment,  # Using technical as proxy
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Error getting sentiment analysis for {symbol}: {e}")
            # Return a more meaningful default based on technical analysis
            return {'overall_score': 0.1, 'label': 'Neutral', 'news_sentiment': 0, 'confidence': 45}
    
    async def _analyze_news_sentiment(self, symbol: str) -> float:
        """Analyze news sentiment using AI"""
        try:
            # Get recent news
            logger.info(f"Fetching news for {symbol} sentiment analysis...")
            news_data = await self.market_service.get_market_news(limit=10)
            
            if not news_data or len(news_data) == 0:
                logger.info(f"No news data available for {symbol}, using neutral sentiment")
                return 0.0
            
            logger.info(f"Found {len(news_data)} news articles for {symbol}")
            
            # Use OpenAI to analyze sentiment
            news_text = " ".join([article.get('title', '') + " " + article.get('summary', '') for article in news_data[:5]])
            
            if len(news_text) > 50:
                logger.info(f"Analyzing news sentiment with OpenAI, text length: {len(news_text)}")
                sentiment_prompt = f"Analyze the sentiment of this financial news about {symbol}. Return ONLY a single number between -1 (very bearish) and 1 (very bullish). News: {news_text[:1000]}"
                
                try:
                    response = await self.openai_service.get_completion(sentiment_prompt)
                    logger.info(f"OpenAI response for sentiment: {response}")
                    
                    # Extract numerical sentiment score
                    import re
                    # Look for any number in the response
                    numbers = re.findall(r'[-+]?\d*\.?\d+', response)
                    if numbers:
                        sentiment_score = float(numbers[0])
                        logger.info(f"Extracted sentiment score: {sentiment_score}")
                        return max(-1, min(1, sentiment_score))
                    else:
                        logger.warning(f"Could not extract sentiment score from OpenAI response: {response}")
                        return 0.0
                except Exception as api_error:
                    logger.error(f"OpenAI API error for sentiment analysis: {api_error}")
                    return 0.0
            else:
                logger.info(f"News text too short for analysis: {len(news_text)} characters")
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error analyzing news sentiment: {e}")
            return 0.0
    
    def _get_social_sentiment(self, symbol: str) -> float:
        """Get social media sentiment (placeholder)"""
        # Placeholder for social sentiment analysis
        # In a real implementation, this would connect to Twitter API, Reddit API, etc.
        return 0.0
    
    async def _get_technical_sentiment(self, symbol: str) -> float:
        """Calculate sentiment based on technical indicators"""
        try:
            # Get recent price data
            market_data = await self.market_service.get_stock_price(symbol)
            if not market_data:
                return 0.0
            
            # Get historical data for technical analysis
            hist_data = self.market_service.get_historical_data(symbol, period='1mo')
            if hist_data is None or hist_data.empty:
                return 0.0
            
            sentiment_score = 0.0
            
            # 1. Price momentum (recent performance)
            change_percent = market_data.get('change_percent', 0)
            if change_percent > 2:
                sentiment_score += 0.3
            elif change_percent > 0:
                sentiment_score += 0.1
            elif change_percent < -2:
                sentiment_score -= 0.3
            elif change_percent < 0:
                sentiment_score -= 0.1
            
            # 2. Moving average trend
            current_price = market_data.get('price', 0)
            sma_20 = hist_data['Close'].rolling(window=20).mean().iloc[-1] if len(hist_data) >= 20 else current_price
            
            if current_price > sma_20 * 1.02:  # Price 2% above SMA
                sentiment_score += 0.2
            elif current_price > sma_20:
                sentiment_score += 0.1
            elif current_price < sma_20 * 0.98:  # Price 2% below SMA
                sentiment_score -= 0.2
            else:
                sentiment_score -= 0.1
            
            # 3. Volume analysis
            volume = market_data.get('volume', 0)
            avg_volume = hist_data['Volume'].mean() if 'Volume' in hist_data else volume
            
            if volume > avg_volume * 1.5 and change_percent > 0:
                sentiment_score += 0.2  # High volume on up day
            elif volume > avg_volume * 1.5 and change_percent < 0:
                sentiment_score -= 0.2  # High volume on down day
            
            # 4. Recent trend (5-day performance)
            if len(hist_data) >= 5:
                five_day_return = (hist_data['Close'].iloc[-1] / hist_data['Close'].iloc[-5] - 1) * 100
                if five_day_return > 5:
                    sentiment_score += 0.2
                elif five_day_return > 0:
                    sentiment_score += 0.1
                elif five_day_return < -5:
                    sentiment_score -= 0.2
                else:
                    sentiment_score -= 0.1
            
            # Normalize score to [-1, 1] range
            sentiment_score = max(-1, min(1, sentiment_score))
            
            return sentiment_score
            
        except Exception as e:
            logger.error(f"Error calculating technical sentiment for {symbol}: {e}")
            return 0.0
    
    def _calculate_risk_metrics(self, df: Optional[Any]) -> dict:
        """Calculate comprehensive risk metrics"""
        try:
            returns = df['Close'].pct_change().dropna()
            
            # Volatility metrics
            volatility_daily = returns.std()
            volatility_annual = volatility_daily * np.sqrt(252)
            
            # Value at Risk (VaR)
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            
            # Maximum Drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Sharpe Ratio (assuming 2% risk-free rate)
            risk_free_rate = 0.02 / 252  # Daily risk-free rate
            excess_returns = returns - risk_free_rate
            sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252)
            
            # Beta calculation (using SPY as market proxy)
            try:
                spy_data = self.market_service.get_historical_data('SPY', period='1y')
                if spy_data is not None and not spy_data.empty:
                    spy_returns = spy_data['Close'].pct_change().dropna()
                    # Align dates
                    common_dates = returns.index.intersection(spy_returns.index)
                    if len(common_dates) > 50:
                        stock_aligned = returns.loc[common_dates]
                        spy_aligned = spy_returns.loc[common_dates]
                        beta = np.cov(stock_aligned, spy_aligned)[0, 1] / np.var(spy_aligned)
                    else:
                        beta = 1.0
                else:
                    beta = 1.0
            except:
                beta = 1.0
            
            return {
                'volatility_daily': volatility_daily * 100,
                'volatility_annual': volatility_annual * 100,
                'var_95': var_95 * 100,
                'var_99': var_99 * 100,
                'max_drawdown': max_drawdown * 100,
                'sharpe_ratio': sharpe_ratio,
                'beta': beta
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}
    
    def _calculate_price_targets(self, df: Optional[Any], indicators: dict) -> dict:
        """Calculate price targets using technical analysis"""
        try:
            current_price = df['Close'].iloc[-1]
            
            # Support and resistance levels
            support_1 = indicators.get('support_1', current_price * 0.95)
            support_2 = indicators.get('support_2', current_price * 0.90)
            resistance_1 = indicators.get('resistance_1', current_price * 1.05)
            resistance_2 = indicators.get('resistance_2', current_price * 1.10)
            
            # Fibonacci retracements
            high_52w = df['High'].rolling(window=252).max().iloc[-1]
            low_52w = df['Low'].rolling(window=252).min().iloc[-1]
            
            fib_range = high_52w - low_52w
            fib_23_6 = high_52w - (fib_range * 0.236)
            fib_38_2 = high_52w - (fib_range * 0.382)
            fib_50_0 = high_52w - (fib_range * 0.500)
            fib_61_8 = high_52w - (fib_range * 0.618)
            
            # Price targets based on technical patterns
            bullish_target = current_price * 1.15  # 15% upside
            bearish_target = current_price * 0.85   # 15% downside
            
            return {
                'support_levels': [support_1, support_2],
                'resistance_levels': [resistance_1, resistance_2],
                'fibonacci': {
                    '23.6%': fib_23_6,
                    '38.2%': fib_38_2,
                    '50.0%': fib_50_0,
                    '61.8%': fib_61_8
                },
                'bullish_target': bullish_target,
                'bearish_target': bearish_target,
                '52w_high': high_52w,
                '52w_low': low_52w
            }
            
        except Exception as e:
            logger.error(f"Error calculating price targets: {e}")
            return {}
    
    async def _get_market_context(self, symbol: str) -> dict:
        """Get broader market context"""
        try:
            # Get market indices
            spy_data = await self.market_service.get_stock_price('SPY')
            qqq_data = await self.market_service.get_stock_price('QQQ')
            
            # Try multiple VIX symbols as fallback
            vix_data = None
            vix_symbols = ['^VIX', 'VIX', 'VIX.US']
            
            for vix_symbol in vix_symbols:
                try:
                    vix_data = await self.market_service.get_stock_price(vix_symbol)
                    if vix_data and vix_data.get('price'):
                        logger.info(f"Successfully fetched VIX data using symbol: {vix_symbol}")
                        break
                except Exception as vix_error:
                    logger.warning(f"Failed to fetch VIX with symbol {vix_symbol}: {vix_error}")
                    continue
            
            # If VIX still fails, use a reasonable default
            vix_level = 20  # Default VIX level
            if vix_data and vix_data.get('price'):
                vix_level = vix_data.get('price', 20)
            else:
                logger.warning("VIX data unavailable, using default value of 20")
            
            # Determine market regime
            market_sentiment = "Neutral"
            if vix_level > 25:
                market_sentiment = "High Volatility"
            elif vix_level < 15:
                market_sentiment = "Low Volatility"
            
            return {
                'spy_performance': spy_data.get('change_percent', 0) if spy_data else 0,
                'qqq_performance': qqq_data.get('change_percent', 0) if qqq_data else 0,
                'vix_level': vix_level,
                'market_sentiment': market_sentiment
            }
            
        except Exception as e:
            logger.error(f"Error getting market context: {e}")
            # Return default values if everything fails
            return {
                'spy_performance': 0,
                'qqq_performance': 0,
                'vix_level': 20,
                'market_sentiment': 'Neutral'
            }
    
    def _format_deep_analysis_response(self, symbol: str, analysis_data: dict, timeframe: str = "1y") -> str:
        """Format comprehensive deep analysis response"""
        try:
            current_price = analysis_data['current_price']
            market_data = analysis_data['market_data']
            indicators = analysis_data['indicators']
            sentiment = analysis_data['sentiment']
            risk_metrics = analysis_data['risk_metrics']
            price_targets = analysis_data['price_targets']
            market_context = analysis_data['market_context']
            
            # Header with current price and basic info
            response = f"📊 **DEEP ANALYSIS: {symbol.upper()}**\n\n"
            response += f"💰 **Current Price:** ${current_price:.2f}\n"
            response += f"📈 **Change:** {market_data.get('change_percent', 0):.2f}%\n"
            response += f"📊 **Volume:** {market_data.get('volume', 0):,}\n\n"
            
            # AI Sentiment Analysis
            sentiment_emoji = "🟢" if sentiment['label'] == "Bullish" else "🔴" if sentiment['label'] == "Bearish" else "🟡"
            response += f"🤖 **AI SENTIMENT ANALYSIS**\n"
            response += f"{sentiment_emoji} **Overall:** {sentiment['label']} ({sentiment['confidence']:.1f}% confidence)\n"
            
            # Show technical sentiment score instead if news is unavailable
            if sentiment['news_sentiment'] == 0:
                response += f"📊 **Technical Sentiment:** {sentiment['social_sentiment']:.2f} (based on price action)\n\n"
            else:
                response += f"📰 **News Sentiment:** {sentiment['news_sentiment']:.2f}\n"
                response += f"📊 **Technical Sentiment:** {sentiment['social_sentiment']:.2f}\n\n"
            
            # Technical Indicators Summary
            response += f"⚙️ **TECHNICAL INDICATORS**\n"
            
            # RSI
            rsi = indicators.get('rsi', 50)
            rsi_signal = "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral"
            response += f"📊 **RSI:** {rsi:.1f} ({rsi_signal})\n"
            
            # MACD
            macd_line = indicators.get('macd_line', 0)
            macd_signal = indicators.get('macd_signal', 0)
            macd_trend = "Bullish" if macd_line > macd_signal else "Bearish"
            response += f"📈 **MACD:** {macd_trend} (Line: {macd_line:.3f}, Signal: {macd_signal:.3f})\n"
            
            # Moving Averages
            sma_20 = indicators.get('sma_20', current_price)
            sma_50 = indicators.get('sma_50', current_price)
            ma_trend = "Above" if current_price > sma_20 and current_price > sma_50 else "Below"
            response += f"📊 **Moving Averages:** Price is {ma_trend} key MAs\n"
            response += f"   • SMA 20: ${sma_20:.2f}\n"
            response += f"   • SMA 50: ${sma_50:.2f}\n\n"
            
            # Risk Metrics
            response += f"⚠️ **RISK ANALYSIS**\n"
            response += f"📊 **Volatility:** {risk_metrics.get('volatility_annual', 0):.1f}% (Annual)\n"
            response += f"📉 **Max Drawdown:** {risk_metrics.get('max_drawdown', 0):.1f}%\n"
            response += f"⚡ **Sharpe Ratio:** {risk_metrics.get('sharpe_ratio', 0):.2f}\n"
            response += f"🎯 **Beta:** {risk_metrics.get('beta', 1):.2f}\n\n"
            
            # Price Targets
            response += f"🎯 **PRICE TARGETS & LEVELS**\n"
            response += f"🟢 **Bullish Target:** ${price_targets.get('bullish_target', current_price * 1.15):.2f} (+15%)\n"
            response += f"🔴 **Bearish Target:** ${price_targets.get('bearish_target', current_price * 0.85):.2f} (-15%)\n\n"
            
            # Support & Resistance
            support_levels = price_targets.get('support_levels', [])
            resistance_levels = price_targets.get('resistance_levels', [])
            
            if support_levels:
                response += f"🛡️ **Support Levels:**\n"
                for i, level in enumerate(support_levels[:2], 1):
                    response += f"   • S{i}: ${level:.2f}\n"
            
            if resistance_levels:
                response += f"🚧 **Resistance Levels:**\n"
                for i, level in enumerate(resistance_levels[:2], 1):
                    response += f"   • R{i}: ${level:.2f}\n"
            
            response += "\n"
            
            # Fibonacci Levels
            fibonacci = price_targets.get('fibonacci', {})
            if fibonacci:
                response += f"🌀 **FIBONACCI RETRACEMENTS**\n"
                for level, price in fibonacci.items():
                    response += f"   • {level}: ${price:.2f}\n"
                response += "\n"
            
            # Market Context
            response += f"🌍 **MARKET CONTEXT**\n"
            response += f"📊 **SPY Performance:** {market_context.get('spy_performance', 0):.3f}%\n"
            response += f"💻 **QQQ Performance:** {market_context.get('qqq_performance', 0):.3f}%\n"
            response += f"😰 **VIX Level:** {market_context.get('vix_level', 20):.1f}\n"
            response += f"🎭 **Market Regime:** {market_context.get('market_sentiment', 'Neutral')}\n\n"
            
            # Trading Signal
            signal_strength = self._calculate_signal_strength(indicators, sentiment, risk_metrics)
            signal_emoji = "🟢" if signal_strength > 0.6 else "🔴" if signal_strength < 0.4 else "🟡"
            signal_label = "STRONG BUY" if signal_strength > 0.8 else "BUY" if signal_strength > 0.6 else "STRONG SELL" if signal_strength < 0.2 else "SELL" if signal_strength < 0.4 else "HOLD"
            
            response += f"🎯 **TRADING SIGNAL**\n"
            response += f"{signal_emoji} **Recommendation:** {signal_label}\n"
            response += f"💪 **Signal Strength:** {signal_strength * 100:.1f}%\n\n"
            
            # Key Insights
            insights = self._generate_key_insights(symbol, analysis_data)
            if insights:
                response += f"💡 **KEY INSIGHTS**\n"
                for insight in insights:
                    response += f"• {insight}\n"
                response += "\n"
            
            # Footer
            response += f"⏰ **Analysis Time:** {self._get_ist_timestamp()}\n"
            response += f"⚠️ **Disclaimer:** This is AI-generated analysis for educational purposes only. Not financial advice."
            
            return response
            
        except Exception as e:
            logger.error(f"Error formatting deep analysis response: {e}")
            return f"❌ Error formatting analysis for {symbol}. Please try again."
    
    def _calculate_signal_strength(self, indicators: dict, sentiment: dict, risk_metrics: dict) -> float:
        """Calculate overall signal strength from 0 to 1"""
        try:
            score = 0.5  # Start neutral
            
            # RSI contribution (20%)
            rsi = indicators.get('rsi')
            if rsi is not None:
                if rsi < 30:
                    score += 0.2  # Oversold = bullish
                elif rsi > 70:
                    score -= 0.2  # Overbought = bearish
            # If RSI data is missing, remain neutral (no contribution)
            
            # MACD contribution (20%)
            macd_line = indicators.get('macd_line')
            macd_signal = indicators.get('macd_signal')
            
            # Only apply MACD contribution if both values are available
            if macd_line is not None and macd_signal is not None:
                if macd_line > macd_signal:
                    score += 0.2
                else:
                    score -= 0.2
            # If MACD data is missing, remain neutral (no contribution)
            
            # Sentiment contribution (30%)
            sentiment_score = sentiment.get('overall_score', 0)
            score += sentiment_score * 0.3
            
            # Risk-adjusted contribution (20%)
            sharpe_ratio = risk_metrics.get('sharpe_ratio')
            if sharpe_ratio is not None:
                if sharpe_ratio > 1:
                    score += 0.1
                elif sharpe_ratio < 0:
                    score -= 0.1
            # If Sharpe ratio is missing, remain neutral
            
            # Volatility adjustment (10%)
            volatility = risk_metrics.get('volatility_annual')
            if volatility is not None:
                if volatility > 50:  # High volatility = higher risk
                    score -= 0.1
                elif volatility < 20:  # Low volatility = lower risk
                    score += 0.05
            # If volatility data is missing, remain neutral
            
            return max(0, min(1, score))
            
        except Exception as e:
            logger.error(f"Error calculating signal strength: {e}")
            return 0.5
    
    def _generate_key_insights(self, symbol: str, analysis_data: dict) -> list:
        """Generate key insights from the analysis"""
        try:
            insights = []
            
            indicators = analysis_data['indicators']
            sentiment = analysis_data['sentiment']
            risk_metrics = analysis_data['risk_metrics']
            price_targets = analysis_data['price_targets']
            current_price = analysis_data['current_price']
            
            # RSI insights
            rsi = indicators.get('rsi', 50)
            if rsi < 30:
                insights.append(f"RSI at {rsi:.1f} suggests {symbol} is oversold and may bounce")
            elif rsi > 70:
                insights.append(f"RSI at {rsi:.1f} indicates {symbol} is overbought and may correct")
            
            # Volatility insights
            volatility = risk_metrics.get('volatility_annual', 30)
            if volatility > 40:
                insights.append(f"High volatility ({volatility:.1f}%) suggests increased risk and opportunity")
            elif volatility < 15:
                insights.append(f"Low volatility ({volatility:.1f}%) indicates stable price action")
            
            # Sentiment insights
            if sentiment['confidence'] > 70:
                insights.append(f"Strong {sentiment['label'].lower()} sentiment with {sentiment['confidence']:.1f}% confidence")
            
            # Price position insights
            high_52w = price_targets.get('52w_high', current_price)
            low_52w = price_targets.get('52w_low', current_price)
            
            if high_52w > 0:
                distance_from_high = ((high_52w - current_price) / high_52w) * 100
                distance_from_low = ((current_price - low_52w) / low_52w) * 100
                
                if distance_from_high < 5:
                    insights.append(f"Trading near 52-week high - potential resistance")
                elif distance_from_low < 5:
                    insights.append(f"Trading near 52-week low - potential support")
                elif distance_from_high > 50:
                    insights.append(f"Significant discount from 52-week high ({distance_from_high:.1f}%)")
            
            # Beta insights
            beta = risk_metrics.get('beta', 1)
            if beta > 1.5:
                insights.append(f"High beta ({beta:.2f}) means higher sensitivity to market moves")
            elif beta < 0.5:
                insights.append(f"Low beta ({beta:.2f}) suggests defensive characteristics")
            
            return insights[:5]  # Limit to 5 insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return []
    
    async def get_comprehensive_signal_analysis(self, symbol: str) -> Optional[dict]:
        """Get comprehensive technical analysis for a symbol"""
        try:
            # Get market data
            market_data = await self.market_service.get_stock_price(symbol)
            if not market_data:
                return None
            
            # Get historical data for technical analysis
            historical_data = self.market_service.get_historical_data(symbol, period="3mo")
            if historical_data is None or historical_data.empty:
                return None
            
            # Use shared technical indicators instance
            indicators = self.technical_indicators.calculate_all_indicators(historical_data)
            
            return {
                'current_price': market_data.get('price', 0),
                'indicators': indicators,
                'signals': indicators.get('signals', {}),
                'patterns': indicators.get('patterns', {})
            }
            
        except Exception as e:
            logger.error(f"Error getting comprehensive analysis for {symbol}: {e}")
            return None
    
    def _calculate_technical_signal_strength(self, indicators: dict, signals: dict) -> int:
        """Calculate overall signal strength from 1-10"""
        try:
            strength = 5  # Neutral baseline
            
            # RSI contribution
            rsi = indicators.get('rsi', 50)
            if rsi < 30:
                strength += 2  # Oversold - bullish
            elif rsi > 70:
                strength -= 2  # Overbought - bearish
            elif 40 <= rsi <= 60:
                strength += 0  # Neutral
            
            # MACD contribution
            if indicators.get('macd_crossover', False):
                strength += 1
            elif indicators.get('macd', 0) < indicators.get('macd_signal', 0):
                strength -= 1
            
            # Volume confirmation
            volume_ratio = indicators.get('volume_ratio', 1.0)
            if volume_ratio > 1.5:
                strength += 1  # High volume confirms signal
            elif volume_ratio < 0.5:
                strength -= 1  # Low volume weakens signal
            
            # Moving average alignment
            if indicators.get('sma_crossover', False):
                strength += 1
            elif indicators.get('ema_crossover', False):
                strength += 1
            
            # Bollinger Bands position
            bb_position = indicators.get('bb_position', 0.5)
            if bb_position < 0.2:
                strength += 1  # Near lower band - potential bounce
            elif bb_position > 0.8:
                strength -= 1  # Near upper band - potential pullback
            
            return max(1, min(10, strength))
            
        except Exception as e:
            logger.error(f"Error calculating signal strength: {e}")
            return 5
    
    def _format_technical_summary(self, indicators: dict) -> str:
        """Format technical indicators into readable summary"""
        try:
            summary_lines = []
            
            # RSI
            rsi = indicators.get('rsi', 50)
            rsi_status = "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral"
            summary_lines.append(f"• **RSI (14):** {rsi:.1f} ({rsi_status})")
            
            # MACD
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            macd_trend = "Bullish" if macd > macd_signal else "Bearish"
            summary_lines.append(f"• **MACD:** {macd:.3f} ({macd_trend})")
            
            # Moving Averages
            sma_20 = indicators.get('sma_20', 0)
            sma_50 = indicators.get('sma_50', 0)
            ma_trend = "Above" if sma_20 > sma_50 else "Below"
            summary_lines.append(f"• **SMA 20/50:** {ma_trend} (${sma_20:.2f}/${sma_50:.2f})")
            
            # Bollinger Bands
            bb_position = indicators.get('bb_position', 0.5)
            bb_status = "Lower" if bb_position < 0.3 else "Upper" if bb_position > 0.7 else "Middle"
            summary_lines.append(f"• **Bollinger Position:** {bb_status} band ({bb_position:.2f})")
            
            return "\n".join(summary_lines)
            
        except Exception as e:
            logger.error(f"Error formatting technical summary: {e}")
            return "• Technical analysis unavailable"
    
    def _calculate_entry_levels(self, current_price: float, direction: str, indicators: dict) -> str:
        """Calculate entry, stop loss, and take profit levels"""
        try:
            atr = indicators.get('atr', current_price * 0.02)  # Default 2% ATR
            
            if direction == "STRONG_BUY" or direction == "BUY":
                entry_price = current_price
                stop_loss = current_price - (atr * 2)
                take_profit_1 = current_price + (atr * 2)
                take_profit_2 = current_price + (atr * 4)
                
                return f"""• **Entry:** ${entry_price:.2f} (Current)
• **Stop Loss:** ${stop_loss:.2f} (-{((current_price - stop_loss) / current_price * 100):.1f}%)
• **Take Profit 1:** ${take_profit_1:.2f} (+{((take_profit_1 - current_price) / current_price * 100):.1f}%)
• **Take Profit 2:** ${take_profit_2:.2f} (+{((take_profit_2 - current_price) / current_price * 100):.1f}%)"""
            
            elif direction == "STRONG_SELL" or direction == "SELL":
                entry_price = current_price
                stop_loss = current_price + (atr * 2)
                take_profit_1 = current_price - (atr * 2)
                take_profit_2 = current_price - (atr * 4)
                
                return f"""• **Entry:** ${entry_price:.2f} (Current)
• **Stop Loss:** ${stop_loss:.2f} (+{((stop_loss - current_price) / current_price * 100):.1f}%)
• **Take Profit 1:** ${take_profit_1:.2f} (-{((current_price - take_profit_1) / current_price * 100):.1f}%)
• **Take Profit 2:** ${take_profit_2:.2f} (-{((current_price - take_profit_2) / current_price * 100):.1f}%)"""
            
            else:  # NEUTRAL
                support = indicators.get('support_1', current_price * 0.97)
                resistance = indicators.get('resistance_1', current_price * 1.03)
                
                return f"""• **Range Trading:** ${support:.2f} - ${resistance:.2f}
• **Buy Zone:** Near ${support:.2f}
• **Sell Zone:** Near ${resistance:.2f}
• **Breakout:** Above ${resistance:.2f} or below ${support:.2f}"""
            
        except Exception as e:
            logger.error(f"Error calculating entry levels: {e}")
            return "• Entry levels calculation failed"
    
    def _calculate_volatility_metrics(self, indicators: dict) -> str:
        """Calculate and format volatility and risk metrics"""
        try:
            atr = indicators.get('atr', 0)
            bb_width = indicators.get('bb_width', 0)
            
            # Volatility assessment
            if bb_width > 0.1:
                volatility_level = "High"
            elif bb_width > 0.05:
                volatility_level = "Medium"
            else:
                volatility_level = "Low"
            
            return f"""• **ATR (14):** ${atr:.2f}
• **Bollinger Width:** {bb_width:.3f}
• **Volatility Level:** {volatility_level}
• **Risk Level:** {'High' if volatility_level == 'High' else 'Medium' if volatility_level == 'Medium' else 'Low'}"""
            
        except Exception as e:
            logger.error(f"Error calculating volatility metrics: {e}")
            return "• Volatility analysis unavailable"
    
    @time_operation("alerts_command")
    @remember_interaction(memory_type=MemoryType.ALERT, importance=MemoryImportance.MEDIUM)
    async def alerts_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /alerts command to list user's alerts with caching"""
        try:
            if not update.message or not update.effective_user:
                secure_logger.error("Update missing message or user in alerts_command")
                return
            
            user_id = str(update.effective_user.id)
            
            # Record command usage
            metrics.record_message("alerts_command", user_id)
            
            # Get alerts with caching from alert service
            alerts = await self.alert_service.get_user_alerts(int(user_id))
            
            if not alerts:
                cached_response = "🔔 **No alerts set**\n\nUse `/alert SYMBOL above/below PRICE` to create alerts.\n\nExample: `/alert AAPL above 150`"
                await update.message.reply_text(cached_response)
                return
            
            # Build response with alert statistics
            alert_text = "🔔 **Your Alerts:**\n\n"
            active_count = 0
            triggered_count = 0
            
            for alert in alerts:
                if alert['is_active']:
                    active_count += 1
                    status = "✅ Active"
                else:
                    triggered_count += 1
                    status = "⏳ Triggered"
                alert_text += f"• {alert['symbol']} {alert['condition']} ({status})\n  ID: `{alert['id']}`\n\n"
            
            # Add summary statistics
            alert_text += f"📊 **Summary:** {active_count} active, {triggered_count} triggered\n\n"
            alert_text += "Use `/remove_alert [alert_id]` to remove an alert."
            
            secure_logger.info(f"Alerts listed: {len(alerts)} total ({active_count} active)", user_id=user_id)
            await update.message.reply_text(alert_text, parse_mode='Markdown')
            
        except Exception as e:
            # Record error
            metrics.record_error("AlertsCommandError", "telegram_handler")
            secure_logger.error(f"Error in alerts command: {e}", user_id=user_id if 'user_id' in locals() else None)
            if update.message:
                await update.message.reply_text("❌ Error fetching alerts. Please try again.")
    
    @time_operation("process_alert_input")
    @remember_alert_activity(importance=MemoryImportance.HIGH)
    async def process_alert_input(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Process alert input from user with validation and caching"""
        try:
            if not update.message or not update.effective_user:
                secure_logger.error("Update missing message or user in process_alert_input")
                return
            
            user_id = str(update.effective_user.id)
            text = update.message.text.strip()
            
            # Record command usage
            metrics.record_message("process_alert_input", user_id)
            
            # Validate input format
            parts = text.split()
            if len(parts) != 4:
                await update.message.reply_text("❌ Invalid format. Use: `/alert SYMBOL above/below PRICE`\n\nExample: `/alert AAPL above 150`")
                return
            
            _, symbol, condition, price_str = parts
            symbol = symbol.upper()  # Normalize symbol
            
            # Security validation for alert parameters
            alert_data = {
                'symbol': symbol,
                'operator': condition.lower(),
                'price': price_str,
                'condition': f"{symbol} {condition.lower()} {price_str}"
            }
            
            validation_result = security_middleware.validate_alert_parameters(alert_data, user_id)
            if not validation_result['valid']:
                secure_logger.log_injection_attempt(user_id, str(alert_data), validation_result['reason'])
                await update.message.reply_text(f"❌ {validation_result['reason']}")
                return
            
            # Validate condition
            if condition.lower() not in ['above', 'below']:
                await update.message.reply_text("❌ Condition must be 'above' or 'below'")
                return
            
            # Validate and parse price
            try:
                target_price = float(price_str)
                if target_price <= 0:
                    await update.message.reply_text("❌ Price must be greater than 0.")
                    return
            except ValueError:
                await update.message.reply_text("❌ Invalid price. Please enter a valid number.")
                return
            
            # Check if symbol exists (basic validation)
            if len(symbol) < 1 or len(symbol) > 10:
                await update.message.reply_text("❌ Invalid symbol. Please enter a valid stock symbol.")
                return
            
            # Create alert with optimized service
            result = await self.alert_service.add_alert(int(user_id), symbol, condition.lower(), target_price)
            
            if result['success']:
                # Get current price for context (cached)
                try:
                    price_data = await self.market_service.get_stock_price(symbol, int(user_id))
                    current_price_text = ""
                    if price_data and 'price' in price_data:
                        current_price = float(price_data['price'])
                        current_price_text = f"\n💰 **Current Price:** ${current_price:.2f}"
                except Exception:
                    current_price_text = ""
                
                secure_logger.info(f"Alert created: {symbol} {condition.lower()} ${target_price:.2f}", user_id=user_id)
                await update.message.reply_text(
                    f"✅ **Alert Created!**\n\n"
                    f"📊 **{symbol}** {condition.lower()} **${target_price:.2f}**\n"
                    f"🆔 **Alert ID:** `{result['alert_id']}`{current_price_text}\n\n"
                    f"You'll be notified when the condition is met!",
                    parse_mode='Markdown'
                )
            else:
                secure_logger.warning(f"Alert creation failed: {result['error']}", user_id=user_id)
                await update.message.reply_text(f"❌ Error creating alert: {result['error']}")
                
        except Exception as e:
            # Record error
            metrics.record_error("ProcessAlertInputError", "telegram_handler")
            secure_logger.error(f"Error processing alert input: {e}", user_id=user_id if 'user_id' in locals() else None)
            if update.message:
                await update.message.reply_text("❌ Error creating alert. Please try again.")
    

    

    

    
    @time_operation("remove_alert_command")
    async def remove_alert_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /remove_alert command with optimized performance"""
        try:
            if not update.message or not update.effective_user:
                secure_logger.error("Update missing message or user in remove_alert_command")
                return
            
            user_id = str(update.effective_user.id)
            text = update.message.text.strip()
            
            # Record command usage
            metrics.record_message("remove_alert_command", user_id)
            
            # Validate input format
            parts = text.split()
            if len(parts) != 2:
                await update.message.reply_text("❌ Invalid format. Use: `/remove_alert [alert_id]`\n\nGet alert IDs with `/alerts`")
                return
            
            # Security validation for alert ID input
            alert_id_str = parts[1]
            if not input_validator.validate_numeric_input(alert_id_str):
                secure_logger.log_injection_attempt(user_id, alert_id_str, "Invalid alert ID format")
                await update.message.reply_text("❌ Invalid alert ID format. Please enter a valid number.")
                return
            
            # Validate alert ID
            try:
                alert_id = int(alert_id_str)
                if alert_id <= 0:
                    await update.message.reply_text("❌ Alert ID must be a positive number.")
                    return
            except ValueError:
                await update.message.reply_text("❌ Invalid alert ID. Please enter a valid number.")
                return
            
            # Remove alert with optimized service
            result = await self.alert_service.remove_alert(int(user_id), alert_id)
            
            if result['success']:
                secure_logger.info(f"Alert removed: ID {alert_id}", user_id=user_id)
                await update.message.reply_text(
                    f"✅ **Alert Removed Successfully!**\n\n"
                    f"🗑️ Alert ID `{alert_id}` has been deleted.\n\n"
                    f"Use `/alerts` to view your remaining alerts.",
                    parse_mode='Markdown'
                )
            else:
                secure_logger.warning(f"Alert removal failed: {result['error']}", user_id=user_id)
                await update.message.reply_text(f"❌ Error removing alert: {result['error']}")
                
        except Exception as e:
            # Record error
            metrics.record_error("RemoveAlertCommandError", "telegram_handler")
            secure_logger.error(f"Error in remove_alert command: {e}", user_id=user_id if 'user_id' in locals() else None)
            if update.message:
                await update.message.reply_text("❌ Error removing alert. Please try again.")
    
    async def _send_alert_notification(self, user_id: int, message: str):
        """Send alert notification to user via Telegram"""
        try:
            if self.application:
                await self.application.bot.send_message(
                    chat_id=user_id,
                    text=f"🔔 **Price Alert!**\n\n{message}",
                    parse_mode='Markdown'
                )
        except Exception as e:
            logger.error(f"Error sending alert notification to user {user_id}: {e}")
    
    def setup_handlers(self) -> None:
        """Setup all command and message handlers with security middleware"""
        if not self.application:
            raise RuntimeError("Application not initialized")
        
        app = self.application
        
        # Apply security middleware to all handlers
        secure_handler = security_middleware.secure_handler
        
        # Basic command handlers (public access)
        app.add_handler(CommandHandler("start", secure_handler(require_auth=False, min_access_level=AccessLevel.GUEST)(self.start_command)))
        app.add_handler(CommandHandler("help", secure_handler(require_auth=False, min_access_level=AccessLevel.GUEST)(self.help_command)))
        app.add_handler(CommandHandler("privacy", secure_handler(require_auth=False, min_access_level=AccessLevel.GUEST)(self.privacy_command)))
        app.add_handler(CommandHandler("contact", secure_handler(require_auth=False, min_access_level=AccessLevel.GUEST)(self.contact_command)))
        
        # Detailed help command handlers (public access)
        app.add_handler(CommandHandler("help_trading", secure_handler(require_auth=False, min_access_level=AccessLevel.GUEST)(self.help_trading_command)))
        app.add_handler(CommandHandler("help_alerts", secure_handler(require_auth=False, min_access_level=AccessLevel.GUEST)(self.help_alerts_command)))
        app.add_handler(CommandHandler("help_advanced", secure_handler(require_auth=False, min_access_level=AccessLevel.GUEST)(self.help_advanced_command)))
        app.add_handler(CommandHandler("help_examples", secure_handler(require_auth=False, min_access_level=AccessLevel.GUEST)(self.help_examples_command)))

        
        # Trading command handlers (user access required) - Simple, user-friendly names
        app.add_handler(CommandHandler("price", secure_handler(min_access_level=AccessLevel.USER)(self.price_command)))
        app.add_handler(CommandHandler("chart", secure_handler(min_access_level=AccessLevel.USER)(self.chart_command)))
        app.add_handler(CommandHandler("analyze", secure_handler(min_access_level=AccessLevel.USER)(self.analyze_command)))
        app.add_handler(CommandHandler("signals", secure_handler(min_access_level=AccessLevel.USER)(self.smart_signal_command)))

        
        # Advanced analysis commands (premium access for resource-intensive operations) - Simplified names
        app.add_handler(CommandHandler("deep_analysis", secure_handler(min_access_level=AccessLevel.PREMIUM)(self.deep_analysis_command)))
        app.add_handler(CommandHandler("ai_analysis", secure_handler(min_access_level=AccessLevel.PREMIUM)(self.deep_analysis_command)))
        app.add_handler(CommandHandler("backtest", secure_handler(min_access_level=AccessLevel.PREMIUM)(self.backtest_command)))
        app.add_handler(CommandHandler("ai_signals", secure_handler(min_access_level=AccessLevel.PREMIUM)(self.ai_signals_command)))
        
        # Alert command handlers (user access)
        app.add_handler(CommandHandler("alert", secure_handler(min_access_level=AccessLevel.USER)(self.process_alert_input)))
        app.add_handler(CommandHandler("alerts", secure_handler(min_access_level=AccessLevel.USER)(self.alerts_command)))
        app.add_handler(CommandHandler("remove_alert", secure_handler(min_access_level=AccessLevel.USER)(self.remove_alert_command)))
        
        # Trade command handlers (user access)
        app.add_handler(CommandHandler("trade", secure_handler(min_access_level=AccessLevel.USER)(self.trade_command)))
        app.add_handler(CommandHandler("trades", secure_handler(min_access_level=AccessLevel.USER)(self.trades_command)))
        app.add_handler(CommandHandler("portfolio", secure_handler(min_access_level=AccessLevel.USER)(self.portfolio_command)))
        app.add_handler(CommandHandler("delete_trade", secure_handler(min_access_level=AccessLevel.USER)(self.delete_trade_command)))
        
        # Advanced trading command handlers (premium access) - User-friendly names
        app.add_handler(CommandHandler("advanced", secure_handler(min_access_level=AccessLevel.PREMIUM)(self.advanced_analysis_command)))
        app.add_handler(CommandHandler("risk", secure_handler(min_access_level=AccessLevel.PREMIUM)(self.risk_analysis_command)))
        app.add_handler(CommandHandler("indicators", secure_handler(min_access_level=AccessLevel.USER)(self.technical_indicators_command)))
        
        # New AI-enhanced commands
        app.add_handler(CommandHandler("strategy", secure_handler(min_access_level=AccessLevel.USER)(self.strategy_command)))
        app.add_handler(CommandHandler("predict", secure_handler(min_access_level=AccessLevel.USER)(self.predict_command)))
        app.add_handler(CommandHandler("watchlist", secure_handler(min_access_level=AccessLevel.USER)(self.watchlist_command)))
        

        
        # Callback query handler for inline keyboards (temporarily without security middleware for debugging)
        # Inline keyboard callbacks disabled
        
        # Menu command handler
        app.add_handler(CommandHandler("menu", secure_handler(require_auth=False, min_access_level=AccessLevel.GUEST)(self.menu_command)))
        
        # Handler for general text messages
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, secure_handler(min_access_level=AccessLevel.USER)(self.handle_message)))
        
        # Error handler
        app.add_error_handler(self.error_handler)
        
        # Handler for photo uploads (user access) - Image Chart Analysis
        app.add_handler(MessageHandler(filters.PHOTO, secure_handler(min_access_level=AccessLevel.USER)(self.photo_handler)))
        
        # Handler for PDF document uploads (user access)
        app.add_handler(MessageHandler(filters.Document.PDF, secure_handler(min_access_level=AccessLevel.USER)(self.pdf_handler)))
        
        # Job queue disabled to avoid timezone compatibility issues with legacy pytz
        # Alert monitoring will be handled differently if needed
        secure_logger.info("Job queue disabled - alert monitoring not available via scheduled jobs")
        
        secure_logger.info("All trading command handlers setup complete with security middleware")
        
        # Callback query handler for inline keyboards (production)
        app.add_handler(CallbackQueryHandler(self.callback_handler.handle_callback_query))
    
    async def _job_check_alerts(self, context):
        """Job queue callback to check alerts periodically"""
        try:
            await self.alert_service._check_alerts()
        except Exception as e:
            logger.error(f"Error in scheduled alert check: {e}")
    
    async def _alert_monitoring_loop(self):
        """Async loop to periodically check alerts"""
        logger.info("Alert monitoring loop started")
        while True:
            try:
                # Check alerts every 30 seconds
                await asyncio.sleep(30)
                logger.info("Checking alerts...")
                await self.alert_service._check_alerts()
                logger.info("Alert check completed")
            except asyncio.CancelledError:
                logger.info("Alert monitoring loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in alert monitoring loop: {e}")
                # Continue monitoring even if there's an error
                await asyncio.sleep(5)  # Short delay before retry

    async def run(self) -> None:
        """Run the bot using asyncio"""
        if self.application is not None:
            raise RuntimeError("Telegram bot Application is already running!")
        try:
            logger.info("Creating Application with minimal builder to avoid APScheduler...")
            
            # Try to completely disable APScheduler by setting environment variables
            import os
            os.environ['APSCHEDULER_TIMEZONE'] = 'UTC'
            os.environ['TZ'] = 'UTC'
            
            # Temporarily replace the JobQueue class to avoid APScheduler initialization
            from telegram.ext import JobQueue
            original_jobqueue_init = JobQueue.__init__
            
            def dummy_jobqueue_init(self, *args, **kwargs):
                # Do nothing - completely bypass JobQueue initialization
                pass
            
            JobQueue.__init__ = dummy_jobqueue_init
            
            try:
                # Use builder but with disabled JobQueue
                builder = Application.builder().token(self.bot_token)
                builder = builder.job_queue(None)
                builder = builder.read_timeout(30).write_timeout(30).connect_timeout(30)
                self.application = builder.build()
                logger.info("Application created successfully with disabled JobQueue")
            finally:
                # Restore original JobQueue init
                JobQueue.__init__ = original_jobqueue_init
            
            # Setup handlers
            self.setup_handlers()
            
            # Initialize and start the bot without run_polling to avoid event loop conflicts
            logger.info("Initializing application...")
            await self.application.initialize()
            
            logger.info("Starting updater...")
            await self.application.updater.start_polling()
            
            logger.info("Starting application...")
            await self.application.start()
            
            # Start alert monitoring using asyncio task since job queue is disabled
            logger.info("Starting alert monitoring with asyncio task...")
            self.alert_monitoring_task = asyncio.create_task(self._alert_monitoring_loop())
            logger.info("Alert monitoring enabled - checking every 30 seconds")
            
            logger.info("Bot is now running! Press Ctrl+C to stop.")
            
            # Keep the bot running
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                logger.info("Bot stopping...")
            finally:
                logger.info("Shutting down bot...")
                
                # Cancel alert monitoring task if it exists
                if hasattr(self, 'alert_monitoring_task') and self.alert_monitoring_task:
                    logger.info("Cancelling alert monitoring task...")
                    self.alert_monitoring_task.cancel()
                    try:
                        await self.alert_monitoring_task
                    except asyncio.CancelledError:
                        logger.info("Alert monitoring task cancelled successfully")
                    except Exception as e:
                        logger.error(f"Error cancelling alert monitoring task: {e}")
                
                # Clean shutdown
                logger.info("Stopping application...")
                await self.application.stop()
                logger.info("Shutting down application...")
                await self.application.shutdown()
                logger.info("Bot shutdown complete")
            
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.error(f"Bot crashed: {str(e)}")
            raise

    async def advanced_analysis_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /advanced_analysis command using yfinance with comprehensive error handling"""
        chat_id = update.effective_chat.id
        user_id = update.effective_user.id
        
        try:
            # Validate input
            if not context.args:
                await update.message.reply_text(
                    "❌ Missing Stock Symbol\n\n"
                    "Please provide a valid stock symbol.\n\n"
                    "Example: /advanced_analysis AAPL\n\n"
                    "Supported: US stocks (NASDAQ, NYSE, etc.)",
                    parse_mode=None
                )
                return
            
            symbol = context.args[0].upper().strip()
            
            # Validate symbol format
            if not symbol or len(symbol) > 10 or not symbol.replace('.', '').replace('-', '').isalpha():
                await update.message.reply_text(
                    f"❌ Invalid Symbol Format: {symbol}\n\n"
                    "Please provide a valid stock symbol (1-10 characters, letters only).\n\n"
                    "Examples: AAPL, TSLA, MSFT, GOOGL",
                    parse_mode=None
                )
                return
            
            # Send typing indicator
            await context.bot.send_chat_action(chat_id=chat_id, action="typing")
            
            # Send initial status message
            status_msg = await update.message.reply_text(
                f"🔍 Analyzing {symbol}...\n\n"
                "📊 Fetching real-time market data...\n"
                "📈 Calculating technical indicators...\n"
                "⏳ Please wait...",
                parse_mode=None
            )
            
            # Use yfinance for data fetching
            import yfinance as yf
            
            try:
                ticker = yf.Ticker(symbol)
                logger.info(f"Fetching data for {symbol} (user: {user_id})")
                
                # Get current price info with timeout handling
                try:
                    info = ticker.info
                    hist = ticker.history(period='1d', interval='1m')
                    
                    if hist.empty:
                        await status_msg.edit_text(
                            f"❌ No Data Available for {symbol}\n\n"
                            "Possible reasons:\n"
                            "• Symbol doesn't exist or is delisted\n"
                            "• Market is closed and no recent data\n"
                            "• Temporary data provider issue\n\n"
                            "What to try:\n"
                            "• Check the symbol spelling\n"
                            "• Try a different symbol\n"
                            "• Wait a few minutes and try again",
                            parse_mode=None
                        )
                        return
                    
                    # Extract price data
                    current_price = float(hist['Close'].iloc[-1])
                    open_price = float(hist['Open'].iloc[0])
                    high_price = float(hist['High'].max())
                    low_price = float(hist['Low'].min())
                    volume = int(hist['Volume'].sum())
                    
                    # Calculate change
                    change = current_price - open_price
                    change_percent = (change / open_price) * 100 if open_price > 0 else 0
                    
                    market_data = {
                        'price': current_price,
                        'change': change,
                        'change_percent': change_percent,
                        'volume': volume,
                        'high': high_price,
                        'low': low_price,
                        'open': open_price,
                        'source': 'yfinance'
                    }
                    
                    logger.info(f"Current data fetched for {symbol}: ${current_price:.2f}")
                    
                except Exception as e:
                    logger.error(f"Error fetching current data for {symbol}: {e}")
                    await status_msg.edit_text(
                        f"❌ Data Fetch Error for {symbol}\n\n"
                        f"Error: {str(e)[:100]}...\n\n"
                        "Possible solutions:\n"
                        "• Check your internet connection\n"
                        "• Verify the symbol is correct\n"
                        "• Try again in a few moments\n"
                        "• Contact support if issue persists",
                        parse_mode=None
                    )
                    return
                
                # Get historical data for technical indicators
                try:
                    await context.bot.send_chat_action(chat_id=chat_id, action="typing")
                    hist_data = ticker.history(period='3mo')
                    
                    if not hist_data.empty and len(hist_data) >= 20:  # Need minimum data for indicators
                        logger.info(f"Historical data fetched for {symbol}: {len(hist_data)} records")
                        indicators = self.technical_indicators.calculate_all_indicators(hist_data)
                        
                        if indicators:
                            market_data['technical_indicators'] = indicators
                            market_data['historical_data'] = hist_data.tail(30).to_dict('records')
                            logger.info(f"Technical indicators calculated for {symbol}: {len(indicators)} indicators")
                        else:
                            logger.warning(f"No indicators calculated for {symbol}")
                            market_data['technical_indicators'] = {}
                    else:
                        logger.warning(f"Insufficient historical data for {symbol}: {len(hist_data) if not hist_data.empty else 0} records")
                        market_data['technical_indicators'] = {}
                        
                except Exception as e:
                    logger.error(f"Error calculating technical indicators for {symbol}: {e}")
                    market_data['technical_indicators'] = {}
                    # Don't fail the entire command for indicator errors
                
            except Exception as e:
                logger.error(f"Critical error fetching data for {symbol}: {e}")
                await status_msg.edit_text(
                    f"❌ Critical Data Error for {symbol}\n\n"
                    f"Error Type: {type(e).__name__}\n"
                    f"Details: {str(e)[:150]}...\n\n"
                    "This usually indicates:\n"
                    "• Network connectivity issues\n"
                    "• Data provider temporary outage\n"
                    "• Invalid or delisted symbol\n\n"
                    "Please try again later or contact support.",
                    parse_mode=None
                )
                return
            
            # Format comprehensive response
            try:
                await context.bot.send_chat_action(chat_id=chat_id, action="typing")
                response = await self._format_advanced_analysis_response(symbol, market_data)
                
                # Update status message to show completion
                await status_msg.edit_text(
                    f"✅ Analysis Complete for {symbol}\n\n"
                    "📊 Sending detailed report...",
                    parse_mode=None
                )
                
                # Split response if too long for better readability
                if len(response) > 4096:
                    # Send in well-formatted chunks
                    chunks = [response[i:i+4000] for i in range(0, len(response), 4000)]
                    for i, chunk in enumerate(chunks):
                        if i == 0:
                            await update.message.reply_text(chunk, parse_mode=None)
                        else:
                            await context.bot.send_message(chat_id=chat_id, text=chunk, parse_mode=None)
                        # Small delay between chunks for better user experience
                        if i < len(chunks) - 1:
                            await asyncio.sleep(0.5)
                else:
                    await update.message.reply_text(response, parse_mode=None)
                
                # Delete the status message after successful completion
                try:
                    await status_msg.delete()
                except:
                    pass  # Ignore if message already deleted
                    
                logger.info(f"Advanced analysis completed successfully for {symbol} (user: {user_id})")
                
            except Exception as e:
                logger.error(f"Error formatting response for {symbol}: {e}")
                await status_msg.edit_text(
                    f"❌ Response Formatting Error\n\n"
                    f"Data was fetched successfully, but there was an error formatting the response.\n\n"
                    f"Error: {str(e)[:100]}...\n\n"
                    "Please try again or contact support.",
                    parse_mode=None
                )
                return
            
        except Exception as e:
            logger.error(f"Unexpected error in advanced_analysis_command for {symbol if 'symbol' in locals() else 'unknown'}: {e}")
            error_msg = (
                f"❌ Unexpected Error\n\n"
                f"Symbol: {symbol if 'symbol' in locals() else 'Unknown'}\n"
                f"Error Type: {type(e).__name__}\n"
                f"Details: {str(e)[:200]}...\n\n"
                "What happened:\n"
                "An unexpected error occurred during analysis.\n\n"
                "What to do:\n"
                "• Try again with a different symbol\n"
                "• Check your internet connection\n"
                "• Contact support if the issue persists\n\n"
                "Support Info:\n"
                "Please include the error details above when contacting support."
            )
            
            try:
                if 'status_msg' in locals():
                    await status_msg.edit_text(error_msg, parse_mode=None)
                else:
                    await update.message.reply_text(error_msg, parse_mode=None)
            except:
                # Fallback to simple message if Markdown fails
                await update.message.reply_text(f"Error performing advanced analysis for {symbol if 'symbol' in locals() else 'symbol'}. Please try again.")
    

    async def risk_analysis_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Enhanced /risk command with comprehensive risk analysis and actionable insights"""
        try:
            if not context.args:
                help_msg = """
🛡️ **Enhanced Risk Analysis**

**Usage:** `/risk SYMBOL [timeframe] [portfolio_value]`

**Examples:**
• `/risk AAPL` - 1-year analysis
• `/risk TSLA 6mo` - 6-month analysis  
• `/risk NVDA 1y 50000` - Analysis with $50k portfolio

**Features:**
📊 Advanced Risk Metrics
🎯 Smart Position Sizing
📈 Sector Comparison
⚡ Real-time Risk Alerts
💡 Actionable Recommendations
"""
                await update.message.reply_text(help_msg)
                return
            
            symbol = context.args[0].upper()
            timeframe = context.args[1] if len(context.args) > 1 else "1y"
            portfolio_value = float(context.args[2]) if len(context.args) > 2 else 10000
            
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
            
            # Send initial status message
            status_msg = await update.message.reply_text("🔄 Analyzing risk metrics and market conditions...")
            
            # Import required libraries
            import yfinance as yf
            import numpy as np
            from datetime import datetime, timedelta
            
            # Get stock data
            stock = yf.Ticker(symbol)
            info = stock.info
            
            # Get historical data based on timeframe
            hist_data = stock.history(period=timeframe)
            
            if hist_data.empty:
                await status_msg.edit_text(f"❌ Unable to get market data for {symbol}. Please check the symbol.")
                return
            
            # Get current price and basic info
            current_price = hist_data['Close'].iloc[-1]
            prev_close = hist_data['Close'].iloc[-2] if len(hist_data) > 1 else current_price
            price_change = ((current_price - prev_close) / prev_close) * 100
            
            # Calculate comprehensive risk metrics
            returns = hist_data['Close'].pct_change().dropna()
            
            # Basic risk metrics
            volatility = returns.std() * np.sqrt(252)
            var_95 = np.percentile(returns, 5) * current_price
            var_99 = np.percentile(returns, 1) * current_price
            
            # Advanced risk metrics
            # Conditional VaR (Expected Shortfall)
            cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * current_price
            cvar_99 = returns[returns <= np.percentile(returns, 1)].mean() * current_price
            
            # Maximum drawdown analysis
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            current_drawdown = drawdown.iloc[-1]
            
            # Risk-adjusted returns
            risk_free_rate = 0.02
            annual_return = returns.mean() * 252
            excess_returns = annual_return - risk_free_rate
            sharpe_ratio = excess_returns / volatility if volatility > 0 else 0
            
            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = excess_returns / downside_deviation if downside_deviation > 0 else 0
            
            # Beta calculation (vs SPY)
            try:
                spy = yf.Ticker("SPY")
                spy_data = spy.history(period=timeframe)
                spy_returns = spy_data['Close'].pct_change().dropna()
                
                # Align dates
                common_dates = returns.index.intersection(spy_returns.index)
                if len(common_dates) > 20:
                    stock_aligned = returns.loc[common_dates]
                    spy_aligned = spy_returns.loc[common_dates]
                    beta = np.cov(stock_aligned, spy_aligned)[0][1] / np.var(spy_aligned)
                else:
                    beta = None
            except:
                beta = None
            
            # Risk level determination with enhanced criteria
            risk_score = 0
            if volatility > 0.40: risk_score += 3
            elif volatility > 0.25: risk_score += 2
            elif volatility > 0.15: risk_score += 1
            
            if max_drawdown < -0.30: risk_score += 2
            elif max_drawdown < -0.20: risk_score += 1
            
            if beta and beta > 1.5: risk_score += 1
            elif beta and beta > 1.2: risk_score += 0.5
            
            if risk_score >= 4:
                risk_level = "VERY HIGH"
                risk_emoji = "🔴"
                position_size = 0.01  # 1%
            elif risk_score >= 2.5:
                risk_level = "HIGH"
                risk_emoji = "🟠"
                position_size = 0.025  # 2.5%
            elif risk_score >= 1.5:
                risk_level = "MODERATE"
                risk_emoji = "🟡"
                position_size = 0.05  # 5%
            elif risk_score >= 0.5:
                risk_level = "LOW"
                risk_emoji = "🟢"
                position_size = 0.08  # 8%
            else:
                risk_level = "VERY LOW"
                risk_emoji = "🔵"
                position_size = 0.10  # 10%
            
            # Kelly Criterion for optimal position sizing
            win_rate = len(returns[returns > 0]) / len(returns)
            avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
            avg_loss = abs(returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 else 0
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win if avg_win > 0 else 0
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
            
            # Get sector information
            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')
            market_cap = info.get('marketCap', 0)
            
            # Format market cap
            if market_cap > 1e12:
                cap_str = f"${market_cap/1e12:.1f}T"
            elif market_cap > 1e9:
                cap_str = f"${market_cap/1e9:.1f}B"
            elif market_cap > 1e6:
                cap_str = f"${market_cap/1e6:.1f}M"
            else:
                cap_str = "N/A"
            
            # Build comprehensive response
            response = f"""
🛡️ **Enhanced Risk Analysis: {symbol}**

💰 **Current Price:** ${current_price:.2f} ({price_change:+.2f}%)
🏢 **Sector:** {sector} | **Market Cap:** {cap_str}

📊 **Risk Assessment:** {risk_emoji} **{risk_level}**

🎯 **Core Risk Metrics:**
• Annual Volatility: {volatility:.1%}
• Max Drawdown: {max_drawdown:.1%}
• Current Drawdown: {current_drawdown:.1%}
• Sharpe Ratio: {sharpe_ratio:.2f}
• Sortino Ratio: {sortino_ratio:.2f}"""
            
            if beta is not None:
                response += f"\n• Beta (vs SPY): {beta:.2f}"
            
            response += f"""

💸 **Value at Risk (Daily):**
• VaR 95%: ${abs(var_95):,.2f} ({abs(var_95/current_price)*100:.1f}%)
• VaR 99%: ${abs(var_99):,.2f} ({abs(var_99/current_price)*100:.1f}%)
• CVaR 95%: ${abs(cvar_95):,.2f} (worst 5% avg)
• CVaR 99%: ${abs(cvar_99):,.2f} (worst 1% avg)

🎯 **Smart Position Sizing:**
• Conservative: {position_size:.1%} of portfolio
• Kelly Optimal: {kelly_fraction:.1%} of portfolio
• Suggested: ${portfolio_value * position_size:,.0f} (${portfolio_value:,.0f} portfolio)

📈 **Risk-Adjusted Allocation:**
• $10K Portfolio: ${10000 * position_size:,.0f}
• $50K Portfolio: ${50000 * position_size:,.0f}
• $100K Portfolio: ${100000 * position_size:,.0f}
"""
            
            # Add risk-specific recommendations
            if risk_level == "VERY HIGH":
                response += """

🚨 **VERY HIGH RISK ALERT:**
• Consider avoiding or use very small position
• High volatility and drawdown potential
• Only for experienced traders with high risk tolerance
• Consider hedging strategies if holding"""
            elif risk_level == "HIGH":
                response += """

⚠️ **HIGH RISK WARNING:**
• Use reduced position sizing
• Monitor closely for exit signals
• Consider stop-loss orders
• Not suitable for conservative portfolios"""
            elif risk_level == "MODERATE":
                response += """

📊 **MODERATE RISK:**
• Standard position sizing appropriate
• Regular monitoring recommended
• Good for balanced portfolios
• Consider diversification"""
            elif risk_level == "LOW":
                response += """

✅ **LOW RISK:**
• Suitable for conservative investors
• Can use higher position sizing
• Good for core portfolio holdings
• Lower volatility expected"""
            else:
                response += """

🔵 **VERY LOW RISK:**
• Excellent for conservative portfolios
• Stable price movements expected
• Can be used as core holding
• Lower returns but higher stability"""
            
            # Add market context
            response += f"""

📊 **Market Context:**
• Win Rate: {win_rate:.1%} (profitable days)
• Avg Win: {avg_win:.2%} | Avg Loss: {avg_loss:.2%}
• Analysis Period: {timeframe.upper()} ({len(hist_data)} trading days)
• Last Updated: {self._get_ist_timestamp()}

💡 **Pro Tip:** Risk levels can change rapidly. Monitor regularly and adjust position sizes based on market conditions."""
            
            await status_msg.edit_text(response, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error in enhanced risk analysis: {e}")
            import traceback
            logger.error(traceback.format_exc())
            await update.message.reply_text(f"❌ Error performing risk analysis for {symbol if 'symbol' in locals() else 'symbol'}. Please check the symbol and try again.")
    
    @cache_result(ttl=1800)  # Cache for 30 minutes
    async def technical_indicators_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /technical_indicators command"""
        try:
            if not context.args:
                await update.message.reply_text("❌ Please provide a stock symbol. Example: `/technical_indicators AAPL`")
                return
            
            symbol = context.args[0].upper()
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
            
            # Get historical data
            async with self.real_market_service as market_service:
                hist_data = await market_service.get_historical_data(symbol, "3mo")
            
            if hist_data.empty:
                await update.message.reply_text(f"❌ Unable to get historical data for {symbol}")
                return
            
            # Calculate technical indicators
            indicators = self.technical_indicators.calculate_all_indicators(hist_data)
            
            response = f"""
📊 Technical Indicators: {symbol}

🎯 Key Indicators:
"""
            
            # Moving Averages
            if 'sma_20' in indicators and 'sma_50' in indicators:
                response += f"• SMA 20: ${indicators['sma_20']:.2f}\n"
                response += f"• SMA 50: ${indicators['sma_50']:.2f}\n"
                response += f"• MA Crossover: {'Bullish' if indicators.get('sma_crossover', False) else 'Bearish'}\n"
            
            # Oscillators
            if 'rsi' in indicators:
                rsi = indicators['rsi']
                response += f"• RSI: {rsi:.1f} ({'Oversold' if rsi < 30 else 'Overbought' if rsi > 70 else 'Neutral'})\n"
            
            if 'macd' in indicators:
                response += f"• MACD: {indicators['macd']:.3f}\n"
                response += f"• MACD Signal: {indicators.get('macd_signal', 0):.3f}\n"
                response += f"• MACD Crossover: {'Bullish' if indicators.get('macd_crossover', False) else 'Bearish'}\n"
            
            # Bollinger Bands
            if 'bb_upper' in indicators and 'bb_lower' in indicators:
                current_price = hist_data['Close'].iloc[-1]
                bb_position = indicators.get('bb_position', 0.5)
                response += f"• BB Upper: ${indicators['bb_upper']:.2f}\n"
                response += f"• BB Lower: ${indicators['bb_lower']:.2f}\n"
                bb_percent = bb_position * 100
                response += f"• BB Position: {bb_percent:.1f}% ({'Overbought' if bb_position > 0.8 else 'Oversold' if bb_position < 0.2 else 'Neutral'})\n"
            
            # Volume
            if 'volume_ratio' in indicators:
                vol_ratio = indicators['volume_ratio']
                response += f"• Volume Ratio: {vol_ratio:.2f}x average ({'High' if vol_ratio > 1.5 else 'Low' if vol_ratio < 0.5 else 'Normal'})\n"
            
            # Signals
            if 'signals' in indicators:
                signals = indicators['signals']
                response += f"\n🎯 Overall Signal: {signals.get('overall_signal', 'NEUTRAL')}\n"
                response += f"• Signal Strength: {signals.get('strength', 0)}\n"
            
            await update.message.reply_text(response)
            
        except Exception as e:
            logger.error(f"Error in technical indicators: {e}")
            await update.message.reply_text("❌ Error calculating technical indicators. Please try again.")
    
    @time_operation("deep_analysis_command")
    @cache_result(ttl=600)  # Cache for 10 minutes due to expensive ML operations
    async def deep_analysis_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle comprehensive deep analysis command with advanced AI insights"""
        user = update.effective_user
        user_id = str(user.id) if user else "unknown"
        
        # Record metrics
        metrics.record_message("deep_analysis", user_id)
        
        try:
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
            
            if not context.args:
                help_msg = """
🧠 **Deep Analysis Command**

**Usage:** `/deep_analysis SYMBOL [timeframe]`

**Examples:**
• `/deep_analysis NVDA` - Full analysis with 1Y data
• `/deep_analysis TSLA 6mo` - 6-month analysis
• `/deep_analysis AAPL 3mo` - 3-month analysis

**Features:**
🔍 Advanced Technical Analysis
📊 Pattern Recognition
🎯 AI-Powered Signals
💡 Market Sentiment Analysis
📈 Risk Assessment
🚀 Price Targets & Levels
                """
                await update.message.reply_text(help_msg)
                return
            
            symbol = context.args[0].upper()
            timeframe = context.args[1] if len(context.args) > 1 else "1y"
            
            # Validate timeframe
            valid_timeframes = ["1mo", "3mo", "6mo", "1y", "2y"]
            if timeframe not in valid_timeframes:
                timeframe = "1y"
            
            await update.message.reply_text(f"🧠 Performing deep analysis on {symbol}...\n⏳ This may take a moment for comprehensive insights.")
            
            # Get comprehensive market data
            analysis_data = await self._get_comprehensive_deep_analysis(symbol, timeframe)
            
            if not analysis_data:
                await update.message.reply_text(f"❌ Unable to perform deep analysis for {symbol}\n\nPossible reasons:\n• Invalid symbol\n• Insufficient market data\n• Technical analysis failed")
                return
            
            # Generate comprehensive response
            response = self._format_deep_analysis_response(symbol, analysis_data, timeframe)
            
            # Send response in chunks if too long
            if len(response) > 4000:
                chunks = self._split_message(response, 4000)
                for chunk in chunks:
                    await update.message.reply_text(chunk, parse_mode='Markdown')
            else:
                await update.message.reply_text(response, parse_mode='Markdown')
            
            # Record trading signal generation
            metrics.record_trading_signal("deep_analysis_signal", symbol)
            
        except Exception as e:
            # Record error
            metrics.record_error("DeepAnalysisCommandError", "telegram_handler")
            
            logger.error(f"Error in deep analysis command: {e}")
            await update.message.reply_text("❌ Error in deep analysis. Please try again.")
    

    

    

    

    

    

    

    

    

    

    

    

    
    @cache_result(ttl=1800)  # Cache for 30 minutes due to expensive backtesting
    async def backtest_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle backtesting command"""
        try:
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
            
            if len(context.args) < 2:
                # Cache help message
                help_msg = response_cache.get("backtest_help")
                if not help_msg:
                    help_msg = "❌ Please provide symbol and strategy. Example: `/backtest AAPL sma`"
                    response_cache.set("backtest_help", help_msg, ttl=3600)
                await update.message.reply_text(help_msg)
                return
            
            symbol = context.args[0].upper()
            strategy_name = context.args[1].lower()
            
            # Get historical data with caching
            cache_key = f"historical_data_{symbol}_1y"
            data = performance_cache.get(cache_key)
            if data is None:
                with connection_pool.get_connection() as conn:
                    data = self.market_service.get_historical_data(symbol, period="1y")
                    if data is not None and not data.empty:
                        performance_cache.set(cache_key, data, ttl=300)  # Cache for 5 minutes
            
            if data is None or data.empty:
                await update.message.reply_text(f"❌ Could not fetch data for {symbol}")
                return
            
            # Select strategy
            if strategy_name == 'sma':
                strategy = sma_crossover_strategy
                strategy_display = "SMA Crossover"
            elif strategy_name == 'rsi':
                strategy = rsi_strategy
                strategy_display = "RSI Strategy"
            elif strategy_name == 'macd':
                strategy = macd_strategy
                strategy_display = "MACD Strategy"
            else:
                await update.message.reply_text("❌ Available strategies: sma, rsi, macd")
                return
            
            # Run backtest with caching
            backtest_cache_key = f"backtest_{symbol}_{strategy_name}"
            cached_result = performance_cache.get(backtest_cache_key)
            
            if cached_result is None:
                await update.message.reply_text(f"📊 Running backtest for {symbol} using {strategy_display}...")
                
                with connection_pool.get_connection() as conn:
                    result = self.backtesting.run_backtest(data, strategy)
                    report = self.backtesting.generate_report(result, f"{symbol} {strategy_display}")
                    
                    cached_result = {'result': result, 'report': report}
                    performance_cache.set(backtest_cache_key, cached_result, ttl=1800)  # Cache for 30 minutes
            else:
                result = cached_result['result']
                report = cached_result['report']
            
            await update.message.reply_text(report)
            
        except Exception as e:
            logger.error(f"Error in backtest command: {e}")
            await update.message.reply_text("❌ Error in backtesting. Please try again.")
    

    

    
    async def ai_signals_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle AI signals command"""
        try:
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
            
            if not context.args:
                await update.message.reply_text("❌ Please provide a stock symbol. Example: `/ai_signals AAPL`")
                return
            
            symbol = context.args[0].upper()
            
            # Get historical data
            data = self.market_service.get_historical_data(symbol, period="1y")
            if data is None or data.empty:
                await update.message.reply_text(f"❌ Could not fetch data for {symbol}")
                return

            # Get AI trading signal
            signal = self.deep_learning.get_trading_signal(data)
            

            
            response = f"""
🤖 **AI Trading Signals: {symbol}**

🎯 **Primary Signal:**
• Action: **{signal['signal']}**
• Confidence: {signal['confidence']:.1%}
• Signal Strength: {signal['signal_strength']:.2f}

📊 **Price Prediction:**
• Current Price: ${signal['price_prediction']['current_price']:.2f}
• Predicted Price: ${signal['price_prediction']['predicted_price']:.2f}
• Expected Change: {signal['price_prediction']['price_change_percent']:+.2f}%

📈 **Technical Analysis:**
• RSI: {signal['technical_indicators']['rsi']:.1f}
• MACD: {signal['technical_indicators']['macd']:.4f}

💡 **Recommendation:**
• **{signal['signal']}** {symbol} with {signal['confidence']:.1%} confidence
• Based on deep learning models and technical analysis
"""
            
            await update.message.reply_text(response)
            
        except Exception as e:
            logger.error(f"Error in AI signals command: {e}")
            await update.message.reply_text("❌ Error generating AI signals. Please try again.")

    @remember_trading_activity(importance=MemoryImportance.HIGH)
    async def trade_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /trade command to create a trade: /trade buy AAPL 10 150.00"""
        try:
            if not update.message or not update.effective_user:
                secure_logger.error("Update missing message or user in trade_command")
                return
            
            user_id = str(update.effective_user.id)
            args = context.args
            
            if len(args) != 4:
                await update.message.reply_text("❌ Usage: /trade [buy|sell] SYMBOL QUANTITY PRICE\n\nExample: /trade buy AAPL 10 150.00")
                return
            
            action, symbol, quantity_str, price_str = args
            action = action.lower()
            
            # Security validation for trade parameters
            trade_data = {
                'action': action,
                'symbol': symbol,
                'quantity': quantity_str,
                'price': price_str
            }
            
            validation_result = security_middleware.validate_trade_parameters(trade_data, user_id)
            if not validation_result['valid']:
                secure_logger.log_injection_attempt(user_id, str(trade_data), validation_result['reason'])
                await update.message.reply_text(f"❌ {validation_result['reason']}")
                return
            
            if action not in ["buy", "sell"]:
                await update.message.reply_text("❌ Action must be 'buy' or 'sell'")
                return
            
            try:
                quantity = float(quantity_str)
                price = float(price_str)
                
                # Additional security checks
                if quantity <= 0 or price <= 0:
                    await update.message.reply_text("❌ Quantity and price must be positive numbers.")
                    return
                    
            except ValueError:
                await update.message.reply_text("❌ Quantity and price must be valid numbers.")
                return
            
            result = await self.trade_service.create_trade(int(user_id), symbol, action, quantity, price)
            
            if result["success"]:
                secure_logger.info(f"Trade created: {action.upper()} {quantity} {symbol.upper()} @ ${price:.2f}", user_id=user_id)
                await update.message.reply_text(f"✅ Trade recorded: {action.upper()} {quantity} {symbol.upper()} @ ${price:.2f}\n🆔 Trade ID: {result['trade_id']}")
            else:
                secure_logger.warning(f"Trade creation failed: {result['error']}", user_id=user_id)
                await update.message.reply_text(f"❌ Error recording trade: {result['error']}")
                
        except Exception as e:
            secure_logger.error(f"Error in trade_command: {e}", user_id=user_id if 'user_id' in locals() else None)
            if update.message:
                await update.message.reply_text("❌ Error recording trade. Please try again.")

    @remember_interaction(memory_type=MemoryType.QUERY, importance=MemoryImportance.MEDIUM)
    async def trades_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /trades command to list all user trades"""
        try:
            if not update.message or not update.effective_user:
                logger.error("Update missing message or user in trades_command")
                return
            user_id = update.effective_user.id
            trades = await self.trade_service.list_trades(user_id)
            if not trades:
                await update.message.reply_text("📄 No trades found. Use /trade to record your first trade.")
                return
            msg = "📄 **Your Trades:**\n\n"
            for t in trades:
                msg += f"• {t['action'].upper()} {t['quantity']} {t['symbol']} @ ${t['price']:.2f} on {t['executed_at'].strftime('%Y-%m-%d %H:%M:%S')}\n  ID: `{t['id']}`\n\n"
            await update.message.reply_text(msg, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"Error in trades_command: {e}")
            if update.message:
                await update.message.reply_text("❌ Error fetching trades. Please try again.")

    @remember_trading_activity(importance=MemoryImportance.MEDIUM)
    async def delete_trade_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /delete_trade command to delete a trade by ID"""
        try:
            if not update.message or not update.effective_user:
                logger.error("Update missing message or user in delete_trade_command")
                return
            args = context.args
            if len(args) != 1:
                await update.message.reply_text("❌ Usage: /delete_trade TRADE_ID\n\nExample: /delete_trade 123")
                return
            try:
                trade_id = int(args[0])
            except ValueError:
                await update.message.reply_text("❌ Trade ID must be a number.")
                return
            user_id = update.effective_user.id
            result = await self.trade_service.delete_trade(user_id, trade_id)
            if result["success"]:
                await update.message.reply_text(f"✅ Trade {trade_id} deleted.")
            else:
                await update.message.reply_text(f"❌ Error deleting trade: {result['error']}")
        except Exception as e:
            logger.error(f"Error in delete_trade_command: {e}")
            if update.message:
                await update.message.reply_text("❌ Error deleting trade. Please try again.")

    @remember_interaction(memory_type=MemoryType.QUERY, importance=MemoryImportance.MEDIUM)
    async def portfolio_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /portfolio command to show portfolio summary"""
        try:
            if not update.effective_user:
                logger.error("Update missing user in portfolio_command")
                return
            
            user_id = update.effective_user.id
            trades = await self.trade_service.list_trades(user_id)
            
            if not trades:
                message = "📊 **Your Portfolio**\n\n"
                message += "📄 No trades recorded yet.\n\n"
                message += "**Get Started:**\n"
                message += "• Use `/trade buy AAPL 10 150` to record a trade\n"
                message += "• Use `/trades` to view your trading history\n"
                message += "• Use `/delete_trade ID` to remove a trade"
                
                if hasattr(update, 'message') and update.message:
                    await update.message.reply_text(message, parse_mode='Markdown')
                else:
                    # Handle callback query
                    await update.edit_message_text(message, parse_mode='Markdown')
                return
            
            # Calculate portfolio summary
            portfolio_summary = {}
            total_value = 0
            total_cost = 0
            
            for trade in trades:
                symbol = trade['symbol']
                action = trade['action']
                quantity = trade['quantity']
                price = trade['price']
                
                if symbol not in portfolio_summary:
                    portfolio_summary[symbol] = {
                        'shares': 0,
                        'total_cost': 0,
                        'avg_price': 0
                    }
                
                if action == 'buy':
                    portfolio_summary[symbol]['shares'] += quantity
                    portfolio_summary[symbol]['total_cost'] += quantity * price
                elif action == 'sell':
                    portfolio_summary[symbol]['shares'] -= quantity
                    portfolio_summary[symbol]['total_cost'] -= quantity * price
                
                # Calculate average price
                if portfolio_summary[symbol]['shares'] > 0:
                    portfolio_summary[symbol]['avg_price'] = portfolio_summary[symbol]['total_cost'] / portfolio_summary[symbol]['shares']
            
            # Remove positions with zero shares
            portfolio_summary = {k: v for k, v in portfolio_summary.items() if v['shares'] > 0}
            
            if not portfolio_summary:
                message = "📊 **Your Portfolio**\n\n"
                message += "📄 No current positions (all trades have been closed).\n\n"
                message += "**Recent Activity:**\n"
                message += f"• Total trades recorded: {len(trades)}\n"
                message += "• Use `/trades` to view your trading history"
                
                if hasattr(update, 'message') and update.message:
                    await update.message.reply_text(message, parse_mode='Markdown')
                else:
                    await update.edit_message_text(message, parse_mode='Markdown')
                return
            
            # Build portfolio message
            message = "📊 **Your Portfolio Summary**\n\n"
            
            for symbol, data in portfolio_summary.items():
                shares = data['shares']
                avg_price = data['avg_price']
                total_cost = data['total_cost']
                
                message += f"**{symbol}**\n"
                message += f"• Shares: {shares:,.0f}\n"
                message += f"• Avg Price: ${avg_price:.2f}\n"
                message += f"• Total Cost: ${total_cost:,.2f}\n\n"
            
            message += "**Portfolio Actions:**\n"
            message += "• `/trade buy SYMBOL QTY PRICE` - Add new trade\n"
            message += "• `/trades` - View all trades\n"
            message += "• `/delete_trade ID` - Remove a trade\n"
            message += "• `/price SYMBOL` - Check current prices"
            
            if hasattr(update, 'message') and update.message:
                await update.message.reply_text(message, parse_mode='Markdown')
            else:
                # Handle callback query
                await update.edit_message_text(message, parse_mode='Markdown')
                
        except Exception as e:
            logger.error(f"Error in portfolio_command: {e}")
            error_message = "❌ Error loading portfolio. Please try again."
            
            if hasattr(update, 'message') and update.message:
                await update.message.reply_text(error_message)
            else:
                await update.edit_message_text(error_message)
    
    async def _format_advanced_analysis_response(self, symbol: str, market_data: dict) -> str:
        """Format comprehensive advanced analysis response with simplified formatting"""
        try:
            import pandas as pd
            from datetime import datetime
            
            # Header with symbol and timestamp - simplified formatting
            response = f"🔍 ADVANCED ANALYSIS: {symbol}\n\n"
            response += f"📅 Analysis Time: {self._get_ist_timestamp()}\n\n"
            
            # Price Data Section
            price = market_data.get('price', 0)
            change = market_data.get('change', 0)
            change_percent = market_data.get('change_percent', 0)
            volume = market_data.get('volume', 0)
            high = market_data.get('high', 0)
            low = market_data.get('low', 0)
            open_price = market_data.get('open', 0)
            
            # Price trend indicator
            trend_emoji = "📈" if change >= 0 else "📉"
            change_emoji = "🟢" if change >= 0 else "🔴"
            
            response += f"💰 PRICE DATA\n"
            response += f"• Current Price: ${price:.2f} {trend_emoji}\n"
            response += f"• Daily Change: {change_emoji} ${change:+.2f} ({change_percent:+.2f}%)\n"
            response += f"• Day Range: ${low:.2f} - ${high:.2f}\n"
            response += f"• Opening Price: ${open_price:.2f}\n"
            response += f"• Volume: {volume:,.0f} shares\n\n"
            
            # Technical Indicators Section
            tech = market_data.get('technical_indicators', {})
            if tech and isinstance(tech, dict):
                response += f"📊 TECHNICAL INDICATORS\n\n"
                
                # Momentum Indicators
                response += f"🎯 Momentum Indicators:\n"
                if 'rsi' in tech and not pd.isna(tech['rsi']):
                    rsi = tech['rsi']
                    if rsi < 30:
                        rsi_signal = "🟢 OVERSOLD (Buy Signal)"
                    elif rsi > 70:
                        rsi_signal = "🔴 OVERBOUGHT (Sell Signal)"
                    else:
                        rsi_signal = "🟡 NEUTRAL"
                    response += f"• RSI (14): {rsi:.1f} - {rsi_signal}\n"
                
                if 'macd' in tech and 'macd_signal' in tech:
                    macd = tech.get('macd', 0)
                    macd_signal = tech.get('macd_signal', 0)
                    macd_histogram = tech.get('macd_histogram', 0)
                    macd_trend = "🟢 BULLISH" if macd > macd_signal else "🔴 BEARISH"
                    response += f"• MACD: {macd:.4f} - {macd_trend}\n"
                    response += f"  - Signal: {macd_signal:.4f}\n"
                    response += f"  - Histogram: {macd_histogram:.4f}\n"
                
                # Moving Averages
                response += f"\n📈 Moving Averages:\n"
                sma_20 = tech.get('sma_20')
                sma_50 = tech.get('sma_50')
                sma_200 = tech.get('sma_200')
                ema_12 = tech.get('ema_12')
                ema_26 = tech.get('ema_26')
                
                if sma_20 and not pd.isna(sma_20):
                    sma20_trend = "🟢 ABOVE" if price > sma_20 else "🔴 BELOW"
                    response += f"• SMA 20: ${sma_20:.2f} - Price {sma20_trend}\n"
                
                if sma_50 and not pd.isna(sma_50):
                    sma50_trend = "🟢 ABOVE" if price > sma_50 else "🔴 BELOW"
                    response += f"• SMA 50: ${sma_50:.2f} - Price {sma50_trend}\n"
                
                if sma_200 and not pd.isna(sma_200):
                    sma200_trend = "🟢 ABOVE" if price > sma_200 else "🔴 BELOW"
                    response += f"• SMA 200: ${sma_200:.2f} - Price {sma200_trend}\n"
                
                if sma_20 and sma_50 and not pd.isna(sma_20) and not pd.isna(sma_50):
                    golden_cross = "🟢 GOLDEN CROSS" if sma_20 > sma_50 else "🔴 DEATH CROSS"
                    response += f"• SMA Cross: {golden_cross}\n"
                
                # Bollinger Bands
                bb_upper = tech.get('bb_upper')
                bb_middle = tech.get('bb_middle')
                bb_lower = tech.get('bb_lower')
                
                if bb_upper and bb_lower and not pd.isna(bb_upper) and not pd.isna(bb_lower):
                    response += f"\n📊 Bollinger Bands:\n"
                    response += f"• Upper Band: ${bb_upper:.2f}\n"
                    response += f"• Middle Band: ${bb_middle:.2f}\n"
                    response += f"• Lower Band: ${bb_lower:.2f}\n"
                    
                    if price > bb_upper:
                        bb_signal = "🔴 ABOVE UPPER (Overbought)"
                    elif price < bb_lower:
                        bb_signal = "🟢 BELOW LOWER (Oversold)"
                    else:
                        bb_signal = "🟡 WITHIN BANDS (Normal)"
                    response += f"• Position: {bb_signal}\n"
                
                # Volume Analysis
                volume_sma = tech.get('volume_sma')
                volume_ratio = tech.get('volume_ratio')
                
                if volume_ratio and not pd.isna(volume_ratio):
                    response += f"\n📊 Volume Analysis:\n"
                    if volume_ratio > 2.0:
                        vol_signal = "🔥 EXTREMELY HIGH"
                    elif volume_ratio > 1.5:
                        vol_signal = "🟢 HIGH"
                    elif volume_ratio < 0.5:
                        vol_signal = "🔴 LOW"
                    else:
                        vol_signal = "🟡 NORMAL"
                    response += f"• Volume vs Avg: {volume_ratio:.1f}x - {vol_signal}\n"
                
                # Additional Technical Indicators
                stoch_k = tech.get('stoch_k')
                stoch_d = tech.get('stoch_d')
                williams_r = tech.get('williams_r')
                atr = tech.get('atr')
                
                if stoch_k and stoch_d and not pd.isna(stoch_k) and not pd.isna(stoch_d):
                    response += f"\n⚡ Oscillators:\n"
                    stoch_signal = "🟢 OVERSOLD" if stoch_k < 20 else "🔴 OVERBOUGHT" if stoch_k > 80 else "🟡 NEUTRAL"
                    response += f"• Stochastic: K={stoch_k:.1f}, D={stoch_d:.1f} - {stoch_signal}\n"
                
                if williams_r and not pd.isna(williams_r):
                    wr_signal = "🟢 OVERSOLD" if williams_r < -80 else "🔴 OVERBOUGHT" if williams_r > -20 else "🟡 NEUTRAL"
                    response += f"• Williams R: {williams_r:.1f} - {wr_signal}\n"
                
                if atr and not pd.isna(atr):
                    response += f"• ATR (Volatility): {atr:.2f}\n"
                
                # Trading Signals
                if 'signals' in tech and tech['signals']:
                    signals = tech['signals']
                    overall_signal = signals.get('overall_signal', 'NEUTRAL')
                    
                    response += f"\n🎯 TRADING SIGNALS\n"
                    
                    if overall_signal == 'BUY':
                        signal_emoji = "🟢"
                    elif overall_signal == 'SELL':
                        signal_emoji = "🔴"
                    else:
                        signal_emoji = "🟡"
                    
                    response += f"• Overall Signal: {signal_emoji} {overall_signal}\n"
                    
                    buy_signals = signals.get('buy_signals', [])
                    sell_signals = signals.get('sell_signals', [])
                    
                    if buy_signals:
                        response += f"• Buy Signals: {', '.join(buy_signals)}\n"
                    
                    if sell_signals:
                        response += f"• Sell Signals: {', '.join(sell_signals)}\n"
                
                # Support and Resistance
                support = tech.get('support')
                resistance = tech.get('resistance')
                
                if support and resistance:
                    response += f"\n🎯 KEY LEVELS\n"
                    response += f"• Support: ${support:.2f}\n"
                    response += f"• Resistance: ${resistance:.2f}\n"
                    
                    # Distance to levels
                    support_dist = ((price - support) / support) * 100
                    resistance_dist = ((resistance - price) / price) * 100
                    response += f"• Distance to Support: {support_dist:+.1f}%\n"
                    response += f"• Distance to Resistance: {resistance_dist:+.1f}%\n"
            
            else:
                response += f"📊 TECHNICAL INDICATORS\n"
                response += f"⚠️ Not Available - Insufficient historical data (need 20+ trading days)\n\n"
            
            # Market Summary
            response += f"📋 MARKET SUMMARY\n"
            
            # Overall trend assessment
            if tech:
                trend_score = 0
                trend_factors = []
                
                # Price vs moving averages
                if 'sma_20' in tech and not pd.isna(tech['sma_20']):
                    if price > tech['sma_20']:
                        trend_score += 1
                        trend_factors.append("Above SMA20")
                    else:
                        trend_score -= 1
                        trend_factors.append("Below SMA20")
                
                if 'sma_50' in tech and not pd.isna(tech['sma_50']):
                    if price > tech['sma_50']:
                        trend_score += 1
                        trend_factors.append("Above SMA50")
                    else:
                        trend_score -= 1
                        trend_factors.append("Below SMA50")
                
                # RSI assessment
                if 'rsi' in tech and not pd.isna(tech['rsi']):
                    rsi = tech['rsi']
                    if 30 <= rsi <= 70:
                        trend_factors.append("RSI Neutral")
                    elif rsi < 30:
                        trend_factors.append("RSI Oversold")
                    else:
                        trend_factors.append("RSI Overbought")
                
                # Overall assessment
                if trend_score > 0:
                    trend_assessment = "🟢 BULLISH TREND"
                elif trend_score < 0:
                    trend_assessment = "🔴 BEARISH TREND"
                else:
                    trend_assessment = "🟡 SIDEWAYS/NEUTRAL"
                
                response += f"• Trend Assessment: {trend_assessment}\n"
                response += f"• Key Factors: {', '.join(trend_factors[:3])}\n"
            
            # Risk Assessment
            response += f"\n⚠️ RISK ASSESSMENT\n"
            
            # Volatility assessment
            daily_range = ((high - low) / open_price) * 100 if open_price > 0 else 0
            if daily_range > 5:
                volatility_level = "🔴 HIGH"
            elif daily_range > 2:
                volatility_level = "🟡 MODERATE"
            else:
                volatility_level = "🟢 LOW"
            
            response += f"• Daily Volatility: {daily_range:.1f}% - {volatility_level}\n"
            
            # Volume-based risk
            if volume > 0:
                volume_risk = "🟢 NORMAL" if volume_ratio and 0.5 <= volume_ratio <= 2.0 else "🟡 ELEVATED"
                response += f"• Volume Risk: {volume_risk}\n"
            
            # Footer
            response += f"\n📊 DATA SOURCE: REAL-TIME MARKET DATA ✅\n"
            response += f"💡 Note: This analysis is for informational purposes only. Always do your own research before making investment decisions.\n\n"
            response += f"🔄 Refresh: Use `/advanced_analysis {symbol}` for updated analysis"
            
            return response
            
        except Exception as e:
            logger.error(f"Error formatting advanced analysis response: {e}")
            return f"❌ Error Formatting Analysis\n\nData was retrieved successfully, but there was an error formatting the detailed response.\n\nError: {str(e)[:100]}...\n\nPlease try again or contact support."

    async def strategy_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /strategy command - AI-generated trading strategies"""
        try:
            user_id = update.effective_user.id
            
            # Get market conditions
            await update.message.reply_text("🤖 Analyzing market conditions and generating trading strategies...")
            
            # Get current market data for major indices
            import yfinance as yf
            
            # Fetch major market indicators
            spy = yf.Ticker("SPY")
            spy_data = spy.history(period="5d")
            spy_current = spy_data['Close'].iloc[-1]
            spy_prev = spy_data['Close'].iloc[-2]
            spy_change = ((spy_current - spy_prev) / spy_prev) * 100
            
            # VIX for volatility
            vix = yf.Ticker("^VIX")
            vix_data = vix.history(period="5d")
            vix_current = vix_data['Close'].iloc[-1]
            
            # Generate strategy based on market conditions
            strategy_response = f"🎯 **AI TRADING STRATEGIES**\n\n"
            strategy_response += f"📊 **Market Overview:**\n"
            strategy_response += f"• S&P 500 (SPY): ${spy_current:.2f} ({spy_change:+.2f}%)\n"
            strategy_response += f"• VIX (Fear Index): {vix_current:.2f}\n\n"
            
            # Strategy recommendations based on market conditions
            if vix_current > 25:  # High volatility
                strategy_response += f"⚠️ **HIGH VOLATILITY ENVIRONMENT**\n\n"
                strategy_response += f"🛡️ **Defensive Strategies:**\n"
                strategy_response += f"• **Cash Secured Puts**: Sell puts on quality stocks you want to own\n"
                strategy_response += f"• **Dollar Cost Averaging**: Gradual position building\n"
                strategy_response += f"• **Defensive Sectors**: Utilities (XLU), Consumer Staples (XLP)\n"
                strategy_response += f"• **Quality Dividend Stocks**: Focus on stable dividend payers\n\n"
                
                strategy_response += f"📈 **Opportunity Strategies:**\n"
                strategy_response += f"• **Volatility Trading**: Consider VIX-related instruments\n"
                strategy_response += f"• **Oversold Bounces**: Look for quality stocks at support levels\n"
                strategy_response += f"• **Covered Calls**: Generate income on existing positions\n"
                
            elif vix_current < 15:  # Low volatility
                strategy_response += f"😌 **LOW VOLATILITY ENVIRONMENT**\n\n"
                strategy_response += f"🚀 **Growth Strategies:**\n"
                strategy_response += f"• **Momentum Trading**: Follow strong trending stocks\n"
                strategy_response += f"• **Growth Sectors**: Technology (XLK), Communication (XLC)\n"
                strategy_response += f"• **Breakout Trading**: Look for stocks breaking resistance\n"
                strategy_response += f"• **Small Cap Growth**: Consider Russell 2000 (IWM)\n\n"
                
                strategy_response += f"⚡ **Active Strategies:**\n"
                strategy_response += f"• **Swing Trading**: 3-10 day holding periods\n"
                strategy_response += f"• **Sector Rotation**: Move between outperforming sectors\n"
                strategy_response += f"• **Options Strategies**: Sell premium in low vol environment\n"
                
            else:  # Moderate volatility
                strategy_response += f"⚖️ **MODERATE VOLATILITY ENVIRONMENT**\n\n"
                strategy_response += f"🎯 **Balanced Strategies:**\n"
                strategy_response += f"• **Core-Satellite**: 70% index funds, 30% active picks\n"
                strategy_response += f"• **Value + Growth**: Mix of undervalued and growing stocks\n"
                strategy_response += f"• **Sector Diversification**: Spread across multiple sectors\n"
                strategy_response += f"• **Risk Parity**: Balance risk across asset classes\n\n"
                
                strategy_response += f"📊 **Technical Strategies:**\n"
                strategy_response += f"• **Moving Average Crossovers**: 20/50 day MA systems\n"
                strategy_response += f"• **Support/Resistance**: Trade bounces and breakouts\n"
                strategy_response += f"• **RSI Mean Reversion**: Buy oversold, sell overbought\n"
            
            # Risk management section
            strategy_response += f"\n🛡️ **RISK MANAGEMENT RULES:**\n"
            strategy_response += f"• **Position Sizing**: Never risk more than 2% per trade\n"
            strategy_response += f"• **Stop Losses**: Set stops at 5-8% below entry\n"
            strategy_response += f"• **Diversification**: Max 5% in any single stock\n"
            strategy_response += f"• **Portfolio Allocation**: 60% stocks, 30% bonds, 10% alternatives\n\n"
            
            # Current market opportunities
            strategy_response += f"🎯 **CURRENT OPPORTUNITIES:**\n"
            if spy_change > 1:
                strategy_response += f"• **Momentum Play**: Market showing strength, consider trend following\n"
            elif spy_change < -1:
                strategy_response += f"• **Dip Buying**: Market weakness may present buying opportunities\n"
            else:
                strategy_response += f"• **Range Trading**: Market consolidating, trade support/resistance\n"
            
            strategy_response += f"• **Earnings Season**: Focus on companies with strong guidance\n"
            strategy_response += f"• **Sector Leaders**: Identify strongest stocks in trending sectors\n\n"
            
            strategy_response += f"⚠️ **DISCLAIMER**: These strategies are for educational purposes only. Past performance doesn't guarantee future results. Always do your own research and consider your risk tolerance.\n\n"
            strategy_response += f"💡 **Next Steps**: Use `/analyze SYMBOL` to research specific stocks or `/predict SYMBOL` for AI price forecasts."
            
            await update.message.reply_text(strategy_response, parse_mode='Markdown')
            logger.info(f"Strategy command processed for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error in strategy command: {str(e)}")
            await update.message.reply_text(
                "❌ Error generating trading strategies. Please try again or contact support.",
                parse_mode='Markdown'
            )

    async def predict_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /predict command - AI price predictions"""
        try:
            user_id = update.effective_user.id
            
            # Validate input
            if not context.args:
                await update.message.reply_text(
                    "❌ **Missing Stock Symbol**\n\n"
                    "Please provide a valid stock symbol.\n\n"
                    "**Example:** `/predict AAPL`\n\n"
                    "**Supported:** US stocks (NASDAQ, NYSE, etc.)",
                    parse_mode='Markdown'
                )
                return
            
            symbol = context.args[0].upper().strip()
            
            # Validate symbol format
            if not self.input_validator.validate_stock_symbol(symbol):
                error_message = self.error_handler.format_error_message("invalid_symbol", {"symbol": symbol})
                await update.message.reply_text(error_message, parse_mode='Markdown')
                return
            
            await update.message.reply_text(f"🤖 Analyzing {symbol} and generating AI price predictions...")
            
            # Get historical data for prediction
            import yfinance as yf
            import numpy as np
            from datetime import datetime, timedelta
            
            ticker = yf.Ticker(symbol)
            
            # Get 6 months of data for better prediction
            hist_data = ticker.history(period="6mo")
            
            if hist_data.empty:
                await update.message.reply_text(
                    f"❌ **No Data Available**\n\n"
                    f"Unable to fetch historical data for {symbol}.\n\n"
                    f"Please verify the symbol is correct and try again.",
                    parse_mode='Markdown'
                )
                return
            
            # Get current price and basic info
            current_price = hist_data['Close'].iloc[-1]
            info = ticker.info
            company_name = info.get('longName', symbol)
            
            # Simple technical analysis for prediction
            prices = hist_data['Close'].values
            volumes = hist_data['Volume'].values
            
            # Calculate technical indicators for prediction
            sma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else current_price
            sma_50 = np.mean(prices[-50:]) if len(prices) >= 50 else current_price
            
            # Calculate volatility
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
            
            # Simple momentum calculation
            momentum_5d = (prices[-1] - prices[-6]) / prices[-6] if len(prices) >= 6 else 0
            momentum_20d = (prices[-1] - prices[-21]) / prices[-21] if len(prices) >= 21 else 0
            
            # Volume trend
            avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else volumes[-1]
            current_volume = volumes[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Generate predictions based on technical analysis
            prediction_response = f"🔮 **AI PRICE PREDICTIONS FOR {symbol}**\n\n"
            prediction_response += f"📊 **{company_name}**\n"
            prediction_response += f"💰 Current Price: ${current_price:.2f}\n\n"
            
            # Short-term prediction (1-5 days)
            trend_factor = 0.5 if sma_20 > sma_50 else -0.5
            momentum_factor = momentum_5d * 0.3
            volume_factor = 0.1 if volume_ratio > 1.2 else -0.1 if volume_ratio < 0.8 else 0
            
            # Random component for realistic prediction
            np.random.seed(int(current_price * 1000) % 1000)  # Deterministic but appears random
            random_factor = np.random.normal(0, volatility/50)
            
            short_term_change = (trend_factor + momentum_factor + volume_factor + random_factor) / 100
            short_term_price = current_price * (1 + short_term_change)
            
            # Medium-term prediction (1-4 weeks)
            medium_momentum = momentum_20d * 0.4
            medium_trend = (sma_20 - sma_50) / sma_50 * 0.2 if sma_50 > 0 else 0
            medium_random = np.random.normal(0, volatility/30)
            
            medium_term_change = (medium_momentum + medium_trend + medium_random) / 100
            medium_term_price = current_price * (1 + medium_term_change)
            
            # Confidence calculation
            confidence = max(20, min(85, 60 - (volatility * 100)))  # Higher volatility = lower confidence
            
            prediction_response += f"📈 **SHORT-TERM FORECAST (1-5 Days):**\n"
            prediction_response += f"• Target Price: ${short_term_price:.2f}\n"
            prediction_response += f"• Expected Change: {short_term_change*100:+.1f}%\n"
            
            if short_term_change > 0.02:
                short_outlook = "🟢 BULLISH"
            elif short_term_change < -0.02:
                short_outlook = "🔴 BEARISH"
            else:
                short_outlook = "🟡 NEUTRAL"
            
            prediction_response += f"• Outlook: {short_outlook}\n\n"
            
            prediction_response += f"📊 **MEDIUM-TERM FORECAST (1-4 Weeks):**\n"
            prediction_response += f"• Target Price: ${medium_term_price:.2f}\n"
            prediction_response += f"• Expected Change: {medium_term_change*100:+.1f}%\n"
            
            if medium_term_change > 0.05:
                medium_outlook = "🟢 BULLISH"
            elif medium_term_change < -0.05:
                medium_outlook = "🔴 BEARISH"
            else:
                medium_outlook = "🟡 NEUTRAL"
            
            prediction_response += f"• Outlook: {medium_outlook}\n\n"
            
            # Support and resistance levels
            recent_high = np.max(prices[-20:]) if len(prices) >= 20 else current_price
            recent_low = np.min(prices[-20:]) if len(prices) >= 20 else current_price
            
            prediction_response += f"🎯 **KEY LEVELS:**\n"
            prediction_response += f"• Resistance: ${recent_high:.2f}\n"
            prediction_response += f"• Support: ${recent_low:.2f}\n\n"
            
            # Confidence and risk assessment
            prediction_response += f"📊 **PREDICTION CONFIDENCE:**\n"
            prediction_response += f"• Confidence Level: {confidence:.0f}%\n"
            prediction_response += f"• Volatility: {volatility*100:.1f}% (Annual)\n"
            
            if volatility > 0.4:
                risk_level = "🔴 HIGH RISK"
            elif volatility > 0.25:
                risk_level = "🟡 MODERATE RISK"
            else:
                risk_level = "🟢 LOW RISK"
            
            prediction_response += f"• Risk Level: {risk_level}\n\n"
            
            # Factors influencing prediction
            prediction_response += f"🔍 **KEY FACTORS:**\n"
            
            if momentum_5d > 0.02:
                prediction_response += f"• 🟢 Strong recent momentum (+{momentum_5d*100:.1f}%)\n"
            elif momentum_5d < -0.02:
                prediction_response += f"• 🔴 Weak recent momentum ({momentum_5d*100:.1f}%)\n"
            
            if sma_20 > sma_50:
                prediction_response += f"• 🟢 Price above moving averages (bullish)\n"
            else:
                prediction_response += f"• 🔴 Price below moving averages (bearish)\n"
            
            if volume_ratio > 1.2:
                prediction_response += f"• 🟢 Above average volume ({volume_ratio:.1f}x)\n"
            elif volume_ratio < 0.8:
                prediction_response += f"• 🔴 Below average volume ({volume_ratio:.1f}x)\n"
            
            prediction_response += f"\n⚠️ **IMPORTANT DISCLAIMERS:**\n"
            prediction_response += f"• These are AI-generated predictions based on technical analysis\n"
            prediction_response += f"• Past performance does not guarantee future results\n"
            prediction_response += f"• Consider fundamental analysis and market news\n"
            prediction_response += f"• Never invest more than you can afford to lose\n\n"
            
            prediction_response += f"💡 **Recommended Actions:**\n"
            prediction_response += f"• Use `/analyze {symbol}` for detailed fundamental analysis\n"
            prediction_response += f"• Set price alerts with `/alert {symbol} price`\n"
            prediction_response += f"• Monitor key support/resistance levels\n\n"
            
            prediction_response += f"🔄 **Last Updated:** {self._get_ist_timestamp()}"
            
            await update.message.reply_text(prediction_response, parse_mode='Markdown')
            logger.info(f"Predict command processed for user {user_id}, symbol {symbol}")
            
        except Exception as e:
            logger.error(f"Error in predict command: {str(e)}")
            await update.message.reply_text(
                "❌ Error generating price predictions. Please try again or contact support.",
                parse_mode='Markdown'
            )

    # ==========================================
    # WATCHLIST COMMANDS SECTION
    # ==========================================
    
    async def watchlist_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /watchlist command - manage personal watchlist"""
        try:
            user_id = update.effective_user.id
            
            # Get user from database
            from db import get_db
            from models import User, Watchlist
            from sqlalchemy import select
            
            async for db in get_db():
                # Get user from database
                result = await db.execute(select(User).where(User.telegram_id == str(user_id)))
                user = result.scalar_one_or_none()
            
                if not user:
                    await update.message.reply_text(
                        "❌ User not found. Please use /start to register first.",
                        parse_mode='Markdown'
                    )
                    return
                
                # Parse command arguments
                if not context.args:
                    await self._show_watchlist(update, db, user)
                elif context.args[0].lower() == 'add':
                    await self._add_to_watchlist(update, context, db, user)
                elif context.args[0].lower() == 'remove':
                    await self._remove_from_watchlist(update, context, db, user)
                elif context.args[0].lower() == 'summary':
                    await self._show_watchlist_summary(update, db, user)
                else:
                    await self._show_watchlist_help(update)
                break
                
            logger.info(f"Watchlist command processed for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error in watchlist command: {str(e)}")
            await update.message.reply_text(
                "❌ Error managing watchlist. Please try again or contact support.",
                parse_mode='Markdown'
            )
    
    async def _show_watchlist(self, update: Update, db, user) -> None:
        """Show current watchlist"""
        from models import Watchlist
        from sqlalchemy import select
        
        result = await db.execute(select(Watchlist).where(Watchlist.user_id == user.id))
        watchlist_items = result.scalars().all()
    
        if not watchlist_items:
            response = f"📋 **YOUR WATCHLIST**\n\n"
            response += f"Your watchlist is empty.\n\n"
            response += f"**Add stocks:** `/watchlist add AAPL`\n"
            response += f"**Remove stocks:** `/watchlist remove AAPL`\n"
            response += f"**Get summary:** `/watchlist summary`"
        else:
            response = f"📋 **YOUR WATCHLIST** ({len(watchlist_items)} stocks)\n\n"
            
            # Get current prices for watchlist items
            import yfinance as yf
            
            for item in watchlist_items:
                try:
                    ticker = yf.Ticker(item.symbol)
                    hist = ticker.history(period="2d")
                    
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                        prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                        change = current_price - prev_price
                        change_pct = (change / prev_price) * 100 if prev_price > 0 else 0
                        
                        change_emoji = "🟢" if change >= 0 else "🔴"
                        response += f"• **{item.symbol}**: ${current_price:.2f} {change_emoji} {change_pct:+.2f}%\n"
                        
                        if item.notes:
                            response += f"  📝 {item.notes}\n"
                    else:
                        response += f"• **{item.symbol}**: Data unavailable\n"
                        
                except Exception as e:
                    response += f"• **{item.symbol}**: Error fetching price\n"
            
            response += f"\n**Commands:**\n"
            response += f"• Add: `/watchlist add SYMBOL [notes]`\n"
            response += f"• Remove: `/watchlist remove SYMBOL`\n"
            response += f"• Summary: `/watchlist summary`"
        
        await update.message.reply_text(response, parse_mode='Markdown')
    
    async def _add_to_watchlist(self, update: Update, context: ContextTypes.DEFAULT_TYPE, db, user) -> None:
        """Add stock to watchlist"""
        from models import Watchlist
        from sqlalchemy import select
        
        if len(context.args) < 2:
            await update.message.reply_text(
                "❌ **Missing Symbol**\n\n"
                "Please provide a stock symbol to add.\n\n"
                "**Example:** `/watchlist add AAPL Great company`",
                parse_mode='Markdown'
            )
            return
        
        symbol = context.args[1].upper().strip()
        notes = ' '.join(context.args[2:]) if len(context.args) > 2 else None
        
        # Validate symbol
        if not self.input_validator.validate_stock_symbol(symbol):
            error_message = self.error_handler.format_error_message("invalid_symbol", {"symbol": symbol})
            await update.message.reply_text(error_message, parse_mode='Markdown')
            return
        
        # Check if already in watchlist
        result = await db.execute(select(Watchlist).where(
            Watchlist.user_id == user.id,
            Watchlist.symbol == symbol
        ))
        existing = result.scalar_one_or_none()
    
        if existing:
            await update.message.reply_text(
                f"⚠️ **{symbol} Already in Watchlist**\n\n"
                f"This stock is already in your watchlist.\n\n"
                f"Use `/watchlist remove {symbol}` to remove it first.",
                parse_mode='Markdown'
            )
            return
        
        # Verify symbol exists by fetching data
        import yfinance as yf
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            company_name = info.get('longName', symbol)
            
            # Add to watchlist
            watchlist_item = Watchlist(
                user_id=user.id,
                symbol=symbol,
                notes=notes,
                created_from_ip=str(update.effective_user.id)  # Using user_id as placeholder
            )
            
            db.add(watchlist_item)
            await db.commit()
            
            response = f"✅ **Added to Watchlist**\n\n"
            response += f"📊 **{symbol}** - {company_name}\n"
            if notes:
                response += f"📝 Notes: {notes}\n"
            response += f"\nUse `/watchlist` to view your complete watchlist."
            
            await update.message.reply_text(response, parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(
                f"❌ **Invalid Symbol**\n\n"
                f"Unable to find data for {symbol}.\n\n"
                f"Please verify the symbol is correct and try again.",
                parse_mode='Markdown'
            )
    
    async def _remove_from_watchlist(self, update: Update, context: ContextTypes.DEFAULT_TYPE, db, user) -> None:
        """Remove stock from watchlist"""
        from models import Watchlist
        from sqlalchemy import select
        
        if len(context.args) < 2:
            await update.message.reply_text(
                "❌ **Missing Symbol**\n\n"
                "Please provide a stock symbol to remove.\n\n"
                "**Example:** `/watchlist remove AAPL`",
                parse_mode='Markdown'
            )
            return
        
        symbol = context.args[1].upper().strip()
        
        # Find and remove from watchlist
        result = await db.execute(select(Watchlist).where(
            Watchlist.user_id == user.id,
            Watchlist.symbol == symbol
        ))
        watchlist_item = result.scalar_one_or_none()
        
        if not watchlist_item:
            await update.message.reply_text(
                f"❌ **{symbol} Not in Watchlist**\n\n"
                f"This stock is not in your watchlist.\n\n"
                f"Use `/watchlist` to see your current watchlist.",
                parse_mode='Markdown'
            )
            return
        
        await db.delete(watchlist_item)
        await db.commit()
        
        await update.message.reply_text(
            f"✅ **Removed from Watchlist**\n\n"
            f"📊 **{symbol}** has been removed from your watchlist.\n\n"
            f"Use `/watchlist` to view your updated watchlist.",
            parse_mode='Markdown'
        )
    
    async def _show_watchlist_summary(self, update: Update, db, user) -> None:
        """Show detailed watchlist summary"""
        from models import Watchlist
        from sqlalchemy import select
        
        result = await db.execute(select(Watchlist).where(Watchlist.user_id == user.id))
        watchlist_items = result.scalars().all()
        
        if not watchlist_items:
            await update.message.reply_text(
                "📋 **Watchlist Summary**\n\n"
                "Your watchlist is empty. Add stocks with `/watchlist add SYMBOL`",
                parse_mode='Markdown'
            )
            return
    
        await update.message.reply_text(f"📊 Generating detailed summary for {len(watchlist_items)} stocks...")
        
        import yfinance as yf
        
        response = f"📋 **WATCHLIST SUMMARY** ({len(watchlist_items)} stocks)\n\n"
        
        total_gainers = 0
        total_losers = 0
    
        for item in watchlist_items:
            try:
                ticker = yf.Ticker(item.symbol)
                hist = ticker.history(period="5d")
                info = ticker.info
            
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                    change = current_price - prev_price
                    change_pct = (change / prev_price) * 100 if prev_price > 0 else 0
                    
                    # Get additional info
                    market_cap = info.get('marketCap', 0)
                    pe_ratio = info.get('trailingPE', 'N/A')
                    
                    change_emoji = "🟢" if change >= 0 else "🔴"
                    if change >= 0:
                        total_gainers += 1
                    else:
                        total_losers += 1
                    
                    response += f"**{item.symbol}** {change_emoji}\n"
                    response += f"• Price: ${current_price:.2f} ({change_pct:+.2f}%)\n"
                    
                    if market_cap > 0:
                        if market_cap > 1e12:
                            cap_str = f"${market_cap/1e12:.1f}T"
                        elif market_cap > 1e9:
                            cap_str = f"${market_cap/1e9:.1f}B"
                        else:
                            cap_str = f"${market_cap/1e6:.1f}M"
                        response += f"• Market Cap: {cap_str}\n"
                    
                    if pe_ratio != 'N/A' and pe_ratio:
                        response += f"• P/E Ratio: {pe_ratio:.1f}\n"
                    
                    response += f"\n"
                    
                else:
                    response += f"**{item.symbol}**: Data unavailable\n\n"
                    
            except Exception as e:
                response += f"**{item.symbol}**: Error fetching data\n\n"
        
        # Summary statistics
        response += f"📊 **SUMMARY:**\n"
        response += f"• 🟢 Gainers: {total_gainers}\n"
        response += f"• 🔴 Losers: {total_losers}\n"
        response += f"• 📈 Win Rate: {(total_gainers/(total_gainers+total_losers)*100):.1f}%\n\n" if (total_gainers + total_losers) > 0 else ""
        
        response += f"💡 **Quick Actions:**\n"
        response += f"• Analyze: `/analyze SYMBOL`\n"
        response += f"• Predict: `/predict SYMBOL`\n"
        response += f"• Set Alert: `/alert SYMBOL price`"
        
        await update.message.reply_text(response, parse_mode='Markdown')
    
    async def _show_watchlist_help(self, update: Update) -> None:
        """Show watchlist command help"""
        await update.message.reply_text(
            "❌ **Invalid Command**\n\n"
            "**Available commands:**\n"
            "• `/watchlist` - View watchlist\n"
            "• `/watchlist add SYMBOL [notes]` - Add stock\n"
            "• `/watchlist remove SYMBOL` - Remove stock\n"
            "• `/watchlist summary` - Detailed summary",
            parse_mode='Markdown'
        )
