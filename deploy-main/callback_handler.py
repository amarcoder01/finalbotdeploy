#!/usr/bin/env python3
"""
Callback Handler for Modern Telegram Bot UI
Handles all inline keyboard interactions and menu navigation
"""

from telegram import Update
from telegram.ext import ContextTypes
from ui_components import TradingBotUI
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ModernCallbackHandler:
    """Handles all callback queries from inline keyboards"""
    
    def __init__(self, telegram_handler):
        self.telegram_handler = telegram_handler
        self.ui = TradingBotUI()
    
    async def handle_callback_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Main callback query handler"""
        query = update.callback_query
        await query.answer()  # Acknowledge the callback
        
        callback_data = query.data
        user_id = update.effective_user.id
        
        # Enhanced logging for debugging
        logger.info(f"CALLBACK HANDLER: Received callback from user {user_id}: {callback_data}")
        print(f"DEBUG: Callback received - User: {user_id}, Data: {callback_data}")
        
        try:
            # Route callback to appropriate handler
            if callback_data.startswith("menu_") or callback_data == "main_menu":
                await self._handle_menu_callback(query, callback_data)
            elif callback_data.startswith("stock_"):
                await self._handle_stock_callback(query, callback_data, context)
            elif callback_data.startswith("price_"):
                await self._handle_price_callback(query, callback_data)
            elif callback_data.startswith("analysis_"):
                await self._handle_analysis_callback(query, callback_data)
            elif callback_data.startswith("alerts_"):
                await self._handle_alerts_callback(query, callback_data)
            elif callback_data.startswith("help_"):
                await self._handle_help_callback(query, callback_data)
            elif callback_data.startswith("confirm_"):
                await self._handle_confirmation_callback(query, callback_data, context)
            elif callback_data.startswith("settings_"):
                await self._handle_settings_callback(query, callback_data)
            elif callback_data.startswith("quick_"):
                await self._handle_quick_callback(query, callback_data, context)
            elif callback_data.startswith("portfolio_"):
                await self._handle_portfolio_callback(query, callback_data, context)
            elif callback_data.startswith("watchlist_"):
                await self._handle_watchlist_callback(query, callback_data, context)
            elif callback_data == "noop":
                pass  # No operation for pagination display
            elif callback_data == "cancel":
                await self._handle_cancel_callback(query)
            else:
                await query.edit_message_text(
                    f"{TradingBotUI.EMOJIS['error']} Unknown action. Please try again.",
                    reply_markup=TradingBotUI.create_main_menu()
                )
        
        except Exception as e:
            logger.error(f"Error handling callback {callback_data}: {str(e)}")
            await query.edit_message_text(
                f"{TradingBotUI.EMOJIS['error']} An error occurred. Returning to main menu.",
                reply_markup=TradingBotUI.create_main_menu()
            )
    
    async def _handle_menu_callback(self, query, callback_data: str) -> None:
        """Handle main menu navigation"""
        if callback_data == "main_menu":
            menu_type = "main"
        else:
            menu_type = callback_data.replace("menu_", "")
        
        if menu_type == "main":
            await query.edit_message_text(
                TradingBotUI.format_welcome_message(),
                reply_markup=TradingBotUI.create_main_menu(),
                parse_mode='Markdown'
            )
        
        elif menu_type == "price":
            message = f"""
{TradingBotUI.EMOJIS['price']} **Stock Price Center**

{TradingBotUI.EMOJIS['target']} Get real-time prices for any US stock
{TradingBotUI.EMOJIS['fire']} Track trending stocks and market movers
{TradingBotUI.EMOJIS['star']} Manage your personal watchlist

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{TradingBotUI.EMOJIS['info']} **Quick Tip:** Type `/price AAPL` for instant results!
            """
            await query.edit_message_text(
                message,
                reply_markup=TradingBotUI.create_price_menu(),
                parse_mode='Markdown'
            )
        
        elif menu_type == "chart":
            message = f"""
{TradingBotUI.EMOJIS['chart']} **Chart Generation Center**

{TradingBotUI.EMOJIS['diamond']} Professional technical charts with indicators
{TradingBotUI.EMOJIS['target']} Multiple timeframes and analysis tools
{TradingBotUI.EMOJIS['fire']} Real-time data visualization

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{TradingBotUI.EMOJIS['info']} **Quick Tip:** Type `/chart TSLA 1d` to get started!

{TradingBotUI.EMOJIS['rocket']} **Enter a stock symbol to generate chart:**
            """
            await query.edit_message_text(
                message,
                reply_markup=TradingBotUI.create_main_menu(), # Changed from InlineKeyboardMarkup to create_main_menu
                parse_mode='Markdown'
            )
        
        elif menu_type == "analysis":
            await query.edit_message_text(
                f"""
{TradingBotUI.EMOJIS['analysis']} **AI Analysis Center**

{TradingBotUI.EMOJIS['bull']} Advanced AI-powered stock analysis
{TradingBotUI.EMOJIS['chart']} Technical indicators and patterns
{TradingBotUI.EMOJIS['target']} Risk assessment and recommendations
{TradingBotUI.EMOJIS['diamond']} Smart trading signals

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{TradingBotUI.EMOJIS['info']} Choose your analysis type below:
                """,
                reply_markup=TradingBotUI.create_analysis_menu(),
                parse_mode='Markdown'
            )
        
        elif menu_type == "alerts":
            await query.edit_message_text(
                f"""
{TradingBotUI.EMOJIS['alert']} **Price Alerts Center**

{TradingBotUI.EMOJIS['target']} Set custom price alerts for any stock
{TradingBotUI.EMOJIS['fire']} Real-time monitoring and notifications
{TradingBotUI.EMOJIS['diamond']} Smart alert suggestions
{TradingBotUI.EMOJIS['gear']} Customizable alert settings

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{TradingBotUI.EMOJIS['info']} Manage your alerts below:
                """,
                reply_markup=TradingBotUI.create_alerts_menu(),
                parse_mode='Markdown'
            )
        
        elif menu_type == "portfolio":
            message = f"""
{TradingBotUI.EMOJIS['portfolio']} **Portfolio Management**

{TradingBotUI.EMOJIS['chart']} Track your investment performance
{TradingBotUI.EMOJIS['target']} Portfolio optimization suggestions
{TradingBotUI.EMOJIS['diamond']} Risk analysis and diversification

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{TradingBotUI.EMOJIS['info']} **Quick Tip:** Use `/portfolio` command for detailed view!
            """
            await query.edit_message_text(
                message,
                reply_markup=TradingBotUI.create_main_menu(), # Changed from InlineKeyboardMarkup to create_main_menu
                parse_mode='Markdown'
            )
        
        elif menu_type == "movers":
            message = f"""
{TradingBotUI.EMOJIS['fire']} **Market Movers**

{TradingBotUI.EMOJIS['up']} Top gainers and trending stocks
{TradingBotUI.EMOJIS['down']} Biggest losers and market shifts
{TradingBotUI.EMOJIS['target']} Volume leaders and breakouts

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{TradingBotUI.EMOJIS['info']} **Quick Tip:** Use `/movers` command for live data!
            """
            await query.edit_message_text(
                message,
                reply_markup=TradingBotUI.create_main_menu(), # Changed from InlineKeyboardMarkup to create_main_menu
                parse_mode='Markdown'
            )
        
        elif menu_type == "help":
            await query.edit_message_text(
                f"""
{TradingBotUI.EMOJIS['help']} **Help & Support Center**

{TradingBotUI.EMOJIS['rocket']} Quick start guide for new users
{TradingBotUI.EMOJIS['chart']} Comprehensive trading tutorials
{TradingBotUI.EMOJIS['diamond']} Advanced features documentation
{TradingBotUI.EMOJIS['info']} Contact support team

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{TradingBotUI.EMOJIS['target']} Choose a help topic below:
                """,
                reply_markup=TradingBotUI.create_help_menu(),
                parse_mode='Markdown'
            )
        
        elif menu_type == "settings":
            message = f"""
{TradingBotUI.EMOJIS['gear']} **Settings & Preferences**

{TradingBotUI.EMOJIS['info']} Customize your trading bot experience:

• Notification preferences
• Display settings
• Account preferences
• Advanced features

{TradingBotUI.EMOJIS['target']} Choose a category to configure:
            """
            await query.edit_message_text(
                message,
                reply_markup=TradingBotUI.create_settings_keyboard(),
                parse_mode='Markdown'
            )
    
    async def _handle_stock_callback(self, query, callback_data: str, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle stock-specific actions"""
        parts = callback_data.split("_")
        if len(parts) < 3:
            return
        
        action = parts[1]
        symbol = parts[2]
        
        if action == "price":
            # Trigger price command
            context.args = [symbol]
            await self.telegram_handler.price_command(query, context)
        
        elif action == "chart":
            # Trigger chart command
            context.args = [symbol, "1d"]
            await self.telegram_handler.chart_command(query, context)
        
        elif action == "analyze":
            # Trigger analysis command
            context.args = [symbol]
            await self.telegram_handler.analyze_command(query, context)
        
        elif action == "alert":
            await query.edit_message_text(
                f"""
{TradingBotUI.EMOJIS['alert']} **Set Price Alert for {symbol}**

{TradingBotUI.EMOJIS['info']} To set an alert, use this format:
`/alert {symbol} above [price]`
`/alert {symbol} below [price]`

{TradingBotUI.EMOJIS['target']} **Examples:**
• `/alert {symbol} above 150`
• `/alert {symbol} below 100`
                """,
                reply_markup=TradingBotUI.create_main_menu(), # Changed from InlineKeyboardMarkup to create_main_menu
                parse_mode='Markdown'
            )
        
        elif action == "watch":
            await query.edit_message_text(
                f"""
{TradingBotUI.EMOJIS['star']} **Watchlist Feature**

{TradingBotUI.EMOJIS['info']} Watchlist functionality is coming soon!

For now, you can:
• Use `/price {symbol}` for quick price checks
• Set alerts with `/alert {symbol} above/below [price]`
• Bookmark this chat for easy access
                """,
                reply_markup=TradingBotUI.create_main_menu(), # Changed from InlineKeyboardMarkup to create_main_menu
                parse_mode='Markdown'
            )
        
        elif action == "buy":
            await query.edit_message_text(
                f"""
💰 **Buy {symbol}**

{TradingBotUI.EMOJIS['info']} **Trading Information:**

This is a demo trading bot. For actual trading, you would need:
• A connected brokerage account
• Real-time market access
• Proper risk management

{TradingBotUI.EMOJIS['target']} **Current Features:**
• Market analysis with `/analyze {symbol}`
• Price tracking with `/price {symbol}`
• Set alerts with `/alert {symbol} above/below [price]`

{TradingBotUI.EMOJIS['warning']} *Always do your own research before investing*
                """,
                reply_markup=TradingBotUI.create_main_menu(), # Changed from InlineKeyboardMarkup to create_main_menu
                parse_mode='Markdown'
            )
        
        elif action == "sell":
            await query.edit_message_text(
                f"""
💸 **Sell {symbol}**

{TradingBotUI.EMOJIS['info']} **Trading Information:**

This is a demo trading bot. For actual trading, you would need:
• A connected brokerage account
• Real-time market access
• Proper risk management

{TradingBotUI.EMOJIS['target']} **Current Features:**
• Market analysis with `/analyze {symbol}`
• Price tracking with `/price {symbol}`
• Set alerts with `/alert {symbol} above/below [price]`

{TradingBotUI.EMOJIS['warning']} *Always do your own research before investing*
                """,
                reply_markup=TradingBotUI.create_main_menu(), # Changed from InlineKeyboardMarkup to create_main_menu
                parse_mode='Markdown'
            )
    
    async def _handle_price_callback(self, query, callback_data: str) -> None:
        """Handle price menu actions"""
        action = callback_data.replace("price_", "")
        
        if action == "quick":
            await query.edit_message_text(
                f"""
{TradingBotUI.EMOJIS['target']} **Quick Price Check**

{TradingBotUI.EMOJIS['info']} Simply type: `/price [SYMBOL]`

{TradingBotUI.EMOJIS['rocket']} **Popular Stocks:**
• `/price AAPL` - Apple Inc.
• `/price TSLA` - Tesla Inc.
• `/price MSFT` - Microsoft Corp.
• `/price GOOGL` - Alphabet Inc.
• `/price AMZN` - Amazon.com Inc.
• `/price NVDA` - NVIDIA Corp.

{TradingBotUI.EMOJIS['diamond']} **Try any US stock symbol!**
                """,
                reply_markup=TradingBotUI.create_price_menu(), # Changed from InlineKeyboardMarkup to create_price_menu
                parse_mode='Markdown'
            )
        
        elif action in ["watchlist", "trending", "gainers"]:
            feature_name = action.title()
            await query.edit_message_text(
                f"""
{TradingBotUI.EMOJIS['star']} **{feature_name} Feature**

{TradingBotUI.EMOJIS['info']} This feature is coming soon!

For now, you can:
• Use `/price [SYMBOL]` for any stock
• Try `/movers` for market trends
• Set alerts with `/alert [SYMBOL] above/below [price]`
                """,
                reply_markup=TradingBotUI.create_price_menu(), # Changed from InlineKeyboardMarkup to create_price_menu
                parse_mode='Markdown'
            )
    
    async def _handle_analysis_callback(self, query, callback_data: str) -> None:
        """Handle analysis menu actions"""
        action = callback_data.replace("analysis_", "")
        
        if action == "ai":
            await query.edit_message_text(
                f"""
{TradingBotUI.EMOJIS['bull']} **AI Stock Analysis**

{TradingBotUI.EMOJIS['info']} Get comprehensive AI-powered analysis for any stock!

{TradingBotUI.EMOJIS['rocket']} **Usage:** `/analyze [SYMBOL]`

{TradingBotUI.EMOJIS['diamond']} **What you'll get:**
• AI-powered market insights
• Technical analysis summary
• Investment recommendations
• Risk assessment
• Market sentiment analysis

{TradingBotUI.EMOJIS['target']} **Try these examples:**
• `/analyze AAPL`
• `/analyze TSLA`
• `/analyze NVDA`
                """,
                reply_markup=TradingBotUI.create_analysis_menu(), # Changed from InlineKeyboardMarkup to create_analysis_menu
                parse_mode='Markdown'
            )
        
        elif action in ["technical", "risk", "signals"]:
            feature_map = {
                "technical": "Technical Analysis",
                "risk": "Risk Assessment", 
                "signals": "Smart Signals"
            }
            feature_name = feature_map[action]
            
            await query.edit_message_text(
                f"""
{TradingBotUI.EMOJIS['chart']} **{feature_name}**

{TradingBotUI.EMOJIS['info']} Advanced {feature_name.lower()} features are coming soon!

For now, try:
• `/analyze [SYMBOL]` - Comprehensive AI analysis
• `/technical_indicators [SYMBOL]` - Technical indicators
• `/smart_signal [SYMBOL]` - AI trading signals
                """,
                reply_markup=TradingBotUI.create_analysis_menu(), # Changed from InlineKeyboardMarkup to create_analysis_menu
                parse_mode='Markdown'
            )
    
    async def _handle_alerts_callback(self, query, callback_data: str) -> None:
        """Handle alerts menu actions"""
        action = callback_data.replace("alerts_", "")
        
        if action == "create":
            await query.edit_message_text(
                f"""
{TradingBotUI.EMOJIS['alert']} **Create Price Alert**

{TradingBotUI.EMOJIS['info']} **Format:** `/alert [SYMBOL] [above/below] [price]`

{TradingBotUI.EMOJIS['target']} **Examples:**
• `/alert AAPL above 150` - Alert when Apple > $150
• `/alert TSLA below 200` - Alert when Tesla < $200
• `/alert MSFT above 300` - Alert when Microsoft > $300

{TradingBotUI.EMOJIS['diamond']} **Features:**
• Real-time monitoring
• Instant notifications
• Multiple alerts supported
• Works 24/7 during market hours
                """,
                reply_markup=TradingBotUI.create_alerts_menu(), # Changed from InlineKeyboardMarkup to create_alerts_menu
                parse_mode='Markdown'
            )
        
        elif action == "view":
            await query.edit_message_text(
                f"""
{TradingBotUI.EMOJIS['chart']} **View Your Alerts**

{TradingBotUI.EMOJIS['info']} Use `/alerts` to see all your active alerts

{TradingBotUI.EMOJIS['target']} **Alert Management:**
• `/alerts` - List all active alerts
• `/remove_alert [ID]` - Remove specific alert
• Each alert has a unique ID for easy management
                """,
                reply_markup=TradingBotUI.create_alerts_menu(), # Changed from InlineKeyboardMarkup to create_alerts_menu
                parse_mode='Markdown'
            )
        
        elif action in ["settings", "smart"]:
            feature_name = "Alert Settings" if action == "settings" else "Smart Alerts"
            await query.edit_message_text(
                f"""
{TradingBotUI.EMOJIS['settings']} **{feature_name}**

{TradingBotUI.EMOJIS['info']} Advanced {feature_name.lower()} are coming soon!

Current alert features:
• Basic price alerts (above/below)
• Real-time monitoring
• Instant notifications
• Multiple alerts per user
                """,
                reply_markup=TradingBotUI.create_alerts_menu(), # Changed from InlineKeyboardMarkup to create_alerts_menu
                parse_mode='Markdown'
            )
    
    async def _handle_help_callback(self, query, callback_data: str) -> None:
        """Handle help menu actions"""
        action = callback_data.replace("help_", "")
        
        if action == "quickstart":
            await query.edit_message_text(
                f"""
{TradingBotUI.EMOJIS['rocket']} **Quick Start Guide**

{TradingBotUI.EMOJIS['target']} **Essential Commands:**
• `/price AAPL` - Get stock price
• `/chart TSLA 1d` - Generate chart
• `/analyze NVDA` - AI analysis
• `/alert MSFT above 300` - Set alert

{TradingBotUI.EMOJIS['diamond']} **Pro Tips:**
• Use natural language: "What's Tesla's price?"
• Try different analysis commands
• Try `/menu` for interactive navigation
• Use `/help` for complete command list

{TradingBotUI.EMOJIS['fire']} **Ready to start trading smarter!**
                """,
                reply_markup=TradingBotUI.create_help_menu(), # Changed from InlineKeyboardMarkup to create_help_menu
                parse_mode='Markdown'
            )
        
        elif action == "trading":
            await query.edit_message_text(
                TradingBotUI.format_help_section(
                    "Trading Commands",
                    [
                        ("/price AAPL", "Get real-time stock price"),
                        ("/chart TSLA 1d", "Generate technical chart"),
                        ("/analyze NVDA", "AI-powered analysis"),
                        ("/movers", "Top market gainers/losers"),
                        ("/portfolio", "Portfolio summary")
                    ],
                    TradingBotUI.EMOJIS['chart']
                ),
                reply_markup=TradingBotUI.create_help_menu(), # Changed from InlineKeyboardMarkup to create_help_menu
                parse_mode='Markdown'
            )
        
        elif action == "alerts":
            await query.edit_message_text(
                TradingBotUI.format_help_section(
                    "Alert Commands",
                    [
                        ("/alert AAPL above 150", "Set price alert (above)"),
                        ("/alert TSLA below 200", "Set price alert (below)"),
                        ("/alerts", "List all active alerts"),
                        ("/remove_alert [ID]", "Remove specific alert")
                    ],
                    TradingBotUI.EMOJIS['alert']
                ),
                reply_markup=TradingBotUI.create_help_menu(), # Changed from InlineKeyboardMarkup to create_help_menu
                parse_mode='Markdown'
            )
        
        elif action == "advanced":
            await query.edit_message_text(
                TradingBotUI.format_help_section(
                    "Advanced Features",
                    [
                        ("/smart_signal AAPL", "AI trading signals"),
                        ("/technical_indicators TSLA", "Technical analysis"),
                        ("/risk_analysis NVDA", "Risk assessment"),
                        ("/advanced_analysis MSFT", "Comprehensive analysis")
                    ],
                    TradingBotUI.EMOJIS['diamond']
                ),
                reply_markup=TradingBotUI.create_help_menu(), # Changed from InlineKeyboardMarkup to create_help_menu
                parse_mode='Markdown'
            )
        
        elif action == "contact":
            await query.edit_message_text(
                f"""
{TradingBotUI.EMOJIS['info']} **Contact & Support**

{TradingBotUI.EMOJIS['target']} **Email:** amar@vidality.com

{TradingBotUI.EMOJIS['diamond']} **Support Topics:**
• Bug reports and technical issues
• Feature requests and suggestions
• Trading strategy questions
• Account and setup help

{TradingBotUI.EMOJIS['rocket']} **Response Time:** Usually within 24 hours

{TradingBotUI.EMOJIS['fire']} We're here to help you trade smarter!
                """,
                reply_markup=TradingBotUI.create_help_menu(), # Changed from InlineKeyboardMarkup to create_help_menu
                parse_mode='Markdown'
            )
    
    async def _handle_confirmation_callback(self, query, callback_data: str, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle confirmation dialogs"""
        parts = callback_data.split("_")
        if len(parts) < 3:
            return
        
        action = parts[1]
        data = "_".join(parts[2:])
        
        # Handle different confirmation types
        if action == "delete_alert":
            # This would integrate with the actual alert deletion logic
            await query.edit_message_text(
                f"{TradingBotUI.EMOJIS['success']} Alert deleted successfully!",
                reply_markup=TradingBotUI.create_main_menu()
            )
        
        # Add more confirmation handlers as needed
    
    async def _handle_settings_callback(self, query, callback_data: str) -> None:
        """Handle settings menu actions"""
        action = callback_data.replace("settings_", "")
        
        if action == "notifications":
            await query.edit_message_text(
                f"""
{TradingBotUI.EMOJIS['alert']} **Notification Settings**

{TradingBotUI.EMOJIS['info']} Customize your notification preferences:

• Price alert notifications
• Market update notifications
• Analysis report notifications
• Portfolio update notifications

{TradingBotUI.EMOJIS['target']} **Coming Soon:** Advanced notification controls!
                """,
                reply_markup=TradingBotUI.create_settings_keyboard(), # Changed from InlineKeyboardMarkup to create_settings_keyboard
                parse_mode='Markdown'
            )
        
        elif action == "display":
            await query.edit_message_text(
                f"""
{TradingBotUI.EMOJIS['chart']} **Display Settings**

{TradingBotUI.EMOJIS['info']} Customize your display preferences:

• Chart themes and colors
• Price display format
• Time zone settings
• Language preferences

{TradingBotUI.EMOJIS['target']} **Coming Soon:** Full customization options!
                """,
                reply_markup=TradingBotUI.create_settings_keyboard(), # Changed from InlineKeyboardMarkup to create_settings_keyboard
                parse_mode='Markdown'
            )
    
    async def _handle_quick_callback(self, query, callback_data: str, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle quick action callbacks"""
        action = callback_data.replace("quick_", "")
        
        if action in ["aapl", "tsla", "msft", "googl", "amzn", "nvda"]:
            symbol = action.upper()
            context.args = [symbol]
            await self.telegram_handler.price_command(query, context)
        
        elif action == "market_overview":
            await self.telegram_handler.movers_command(query, context)
    
    async def _handle_portfolio_callback(self, query, callback_data: str, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle portfolio menu actions"""
        action = callback_data.replace("portfolio_", "")
        
        if action == "view":
            await self.telegram_handler.portfolio_command(query, context)
        
        elif action == "add":
            await query.edit_message_text(
                f"""
{TradingBotUI.EMOJIS['portfolio']} **Add to Portfolio**

{TradingBotUI.EMOJIS['info']} Use the following format to add holdings:
`/portfolio add [SYMBOL] [SHARES] [PRICE]`

{TradingBotUI.EMOJIS['target']} **Examples:**
• `/portfolio add AAPL 10 150.00`
• `/portfolio add TSLA 5 200.00`
                """,
                reply_markup=TradingBotUI.create_main_menu(), # Changed from InlineKeyboardMarkup to create_main_menu
                parse_mode='Markdown'
            )
    
    async def _handle_watchlist_callback(self, query, callback_data: str, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle watchlist menu actions"""
        action = callback_data.replace("watchlist_", "")
        
        if action == "view":
            await query.edit_message_text(
                f"""
{TradingBotUI.EMOJIS['star']} **Your Watchlist**

{TradingBotUI.EMOJIS['info']} Watchlist feature is coming soon!

For now, you can:
• Use `/price [SYMBOL]` for quick checks
• Set alerts with `/alert [SYMBOL] above/below [price]`
• Bookmark frequently checked stocks
                """,
                reply_markup=TradingBotUI.create_main_menu(), # Changed from InlineKeyboardMarkup to create_main_menu
                parse_mode='Markdown'
            )
        
        elif action == "add":
            await query.edit_message_text(
                f"""
{TradingBotUI.EMOJIS['star']} **Add to Watchlist**

{TradingBotUI.EMOJIS['info']} Watchlist functionality is coming soon!

For now, try:
• `/price [SYMBOL]` - Quick price check
• `/alert [SYMBOL] above/below [price]` - Set price alerts
• Save this chat for easy access to your favorite stocks
                """,
                reply_markup=TradingBotUI.create_main_menu(), # Changed from InlineKeyboardMarkup to create_main_menu
                parse_mode='Markdown'
            )
    
    async def _handle_cancel_callback(self, query) -> None:
        """Handle cancel actions"""
        await query.edit_message_text(
            TradingBotUI.format_welcome_message(),
            reply_markup=TradingBotUI.create_main_menu(),
            parse_mode='Markdown'
        )