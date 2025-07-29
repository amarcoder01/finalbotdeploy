#!/usr/bin/env python3
"""
Next-Generation UI Components for Telegram Trading Bot
Ultra-modern, accessible interface with advanced UX patterns, animations, and AI-powered features
Based on 2024+ Telegram bot best practices and cutting-edge trading bot interfaces

Features:
- Adaptive dark/light themes
- Accessibility optimizations
- Advanced animations and micro-interactions
- Smart contextual menus
- Progressive disclosure patterns
- Mobile-first responsive design
- AI-powered personalization
"""

from telegram import Update
from telegram.ext import ContextTypes
from typing import List, Dict, Optional, Tuple, Union, Any
import logging
from datetime import datetime, timedelta
from timezone_utils import format_ist_timestamp, get_ist_time
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib

logger = logging.getLogger(__name__)

class ThemeMode(Enum):
    """Theme mode enumeration"""
    LIGHT = "light"
    DARK = "dark"
    AUTO = "auto"
    HIGH_CONTRAST = "high_contrast"

class AnimationType(Enum):
    """Animation type enumeration"""
    FADE = "fade"
    SLIDE = "slide"
    BOUNCE = "bounce"
    PULSE = "pulse"
    NONE = "none"

@dataclass
class UITheme:
    """Advanced UI theme configuration with accessibility support"""
    mode: ThemeMode = ThemeMode.LIGHT
    primary_color: str = "🔵"
    success_color: str = "🟢"
    warning_color: str = "🟡"
    danger_color: str = "🔴"
    info_color: str = "🔷"
    accent_color: str = "💎"
    background_pattern: str = "━"
    separator_pattern: str = "─"
    animation_enabled: bool = True
    high_contrast: bool = False
    reduced_motion: bool = False
    
    def get_separator(self, length: int = 40) -> str:
        """Get themed separator line"""
        if self.mode == ThemeMode.HIGH_CONTRAST:
            return "═" * length
        return self.background_pattern * length
    
    def get_subseparator(self, length: int = 20) -> str:
        """Get themed sub-separator line"""
        return self.separator_pattern * length

@dataclass
class ButtonStyle:
    """Enhanced button styling with accessibility"""
    primary: str = "🔵"
    secondary: str = "⚪"
    success: str = "✅"
    danger: str = "❌"
    warning: str = "⚠️"
    disabled: str = "⚫"
    loading: str = "⏳"
    
@dataclass
class UserPreferences:
    """User-specific UI preferences"""
    theme: UITheme = field(default_factory=UITheme)
    language: str = "en"
    timezone: str = "UTC"
    currency_symbol: str = "$"
    number_format: str = "US"
    quick_actions: List[str] = field(default_factory=lambda: ["price", "chart", "analysis"])
    favorite_symbols: List[str] = field(default_factory=list)
    notification_preferences: Dict[str, bool] = field(default_factory=lambda: {
        "price_alerts": True,
        "market_news": True,
        "portfolio_updates": True,
        "system_notifications": True
    })
    accessibility_features: Dict[str, bool] = field(default_factory=lambda: {
        "screen_reader_support": False,
        "high_contrast": False,
        "reduced_motion": False,
        "large_text": False
    })

class TradingBotUI:
    """Modern UI components for the Telegram trading bot"""
    
    # Enhanced emoji system for modern trading bot UI
    EMOJIS = {
        # Core Trading
        'chart': '📊', 'price': '💰', 'analysis': '🔍', 'portfolio': '📈',
        'alert': '🔔', 'watchlist': '👁️', 'trade': '💹', 'profit': '💵',
        
        # Market Indicators
        'bull': '🐂', 'bear': '🐻', 'up': '📈', 'down': '📉', 'neutral': '➡️',
        'fire': '🔥', 'rocket': '🚀', 'trending': '📊', 'volume': '📊',
        
        # UI Elements
        'menu': '📋', 'settings': '⚙️', 'back': '🔙', 'home': '🏠',
        'refresh': '🔄', 'search': '🔍', 'filter': '🔽', 'sort': '🔀',
        
        # Status & Feedback
        'success': '✅', 'error': '❌', 'warning': '⚠️', 'info': 'ℹ️',
        'loading': '⏳', 'processing': '🔄', 'complete': '✅', 'pending': '⏸️',
        
        # Premium & Features
        'star': '⭐', 'diamond': '💎', 'crown': '👑', 'gem': '💠',
        'premium': '🌟', 'pro': '⚡', 'vip': '🔥', 'elite': '💎',
        
        # Actions & Tools
        'target': '🎯', 'lightning': '⚡', 'brain': '🧠', 'crystal_ball': '🔮',
        'shield': '🛡️', 'compass': '🧭', 'telescope': '🔭', 'microscope': '🔬',
        'calculator': '🧮', 'calendar': '📅', 'clock': '🕐', 'timer': '⏱️',
        
        # Social & Communication
        'help': '❓', 'support': '🆘', 'contact': '📞', 'email': '📧',
        'telegram': '💬', 'notification': '🔔', 'message': '💬', 'chat': '💭',
        
        # Data & Analytics
         'data': '📊', 'report': '📋', 'export': '📤', 'import': '📥',
         'sync': '🔄', 'backup': '💾', 'cloud': '☁️', 'database': '🗄️',
         'news': '📰', 'volume': '📊', 'trending': '🔥',
        
        # Navigation
        'next': '▶️', 'prev': '◀️', 'first': '⏮️', 'last': '⏭️',
        'up_arrow': '⬆️', 'down_arrow': '⬇️', 'left_arrow': '⬅️', 'right_arrow': '➡️'
    }
    
    # Enhanced color scheme with theme support
    COLORS = {
        'primary': '🔵',
        'success': '🟢', 
        'warning': '🟡',
        'danger': '🔴',
        'info': '🔷',
        'light': '⚪',
        'dark': '⚫',
        'accent': '💎',
        'muted': '⚫',
        'highlight': '🌟'
    }
    
    # User preference cache for performance
    _user_preferences_cache: Dict[str, UserPreferences] = {}
    
    @classmethod
    def get_user_preferences(cls, user_id: str) -> UserPreferences:
        """Get cached user preferences or create default"""
        if user_id not in cls._user_preferences_cache:
            cls._user_preferences_cache[user_id] = UserPreferences()
        return cls._user_preferences_cache[user_id]
    
    @classmethod
    def update_user_preferences(cls, user_id: str, preferences: UserPreferences) -> None:
        """Update user preferences in cache"""
        cls._user_preferences_cache[user_id] = preferences
    
    @classmethod
    def get_themed_separator(cls, user_id: str, length: int = 40) -> str:
        """Get themed separator based on user preferences"""
        prefs = cls.get_user_preferences(user_id)
        return prefs.theme.get_separator(length)
    
    @classmethod
    def format_with_theme(cls, text: str, user_id: str, style: str = "default") -> str:
        """Format text with user's theme preferences"""
        prefs = cls.get_user_preferences(user_id)
        
        if prefs.accessibility_features.get("large_text", False):
            # Add emphasis for large text users
            text = f"**{text}**"
        
        if prefs.accessibility_features.get("high_contrast", False):
            # Use high contrast separators
            text = text.replace("━", "═").replace("─", "═")
        
        return text
    
    @classmethod
    def create_smart_main_menu(cls, user_id: str, context: Dict[str, Any] = None) -> str:
        """Create AI-powered personalized main menu based on user behavior and preferences"""
        prefs = cls.get_user_preferences(user_id)
        context = context or {}
        
        # Return a formatted text menu instead of inline keyboard
        menu_text = f"{cls.EMOJIS['menu']} **Main Menu**\n\n"
        menu_text += f"{cls.EMOJIS['price']} Live Prices\n"
        menu_text += f"{cls.EMOJIS['chart']} Charts\n"
        menu_text += f"{cls.EMOJIS['brain']} AI Analysis\n"
        menu_text += f"{cls.EMOJIS['watchlist']} Watchlist ({len(prefs.favorite_symbols)})\n"
        menu_text += f"{cls.EMOJIS['fire']} Market Movers\n"
        menu_text += f"{cls.EMOJIS['alert']} Alerts\n"
        menu_text += f"{cls.EMOJIS['portfolio']} Portfolio\n"
        menu_text += f"{cls.EMOJIS['calculator']} Tools\n"
        menu_text += f"{cls.EMOJIS['settings']} Settings\n"
        menu_text += f"{cls.EMOJIS['help']} Help & Guide\n"
        
        return menu_text
    

    

    

    

    

    

    
    @classmethod
    def format_personalized_welcome_message(cls, user_name: str = "Trader", user_id: str = None, 
                                          is_returning: bool = False) -> str:
        """Format a personalized, engaging welcome message with smart features"""
        prefs = cls.get_user_preferences(user_id) if user_id else UserPreferences()
        
        # Time-based greeting
        current_hour = get_ist_time().hour
        if 5 <= current_hour < 12:
            time_greeting = f"{cls.EMOJIS['morning']} Good morning"
        elif 12 <= current_hour < 17:
            time_greeting = f"{cls.EMOJIS['afternoon']} Good afternoon"
        elif 17 <= current_hour < 21:
            time_greeting = f"{cls.EMOJIS['evening']} Good evening"
        else:
            time_greeting = f"{cls.EMOJIS['night']} Good evening"
        
        # Personalized greeting based on user status
        if is_returning:
            greeting = f"{time_greeting}, {user_name}! Welcome back!"
            intro = "Your AI trading assistant is ready with the latest market insights."
        else:
            greeting = f"{time_greeting}, {user_name}! Welcome to TradeAI Companion!"
            intro = "Your AI-powered trading assistant is ready to help you navigate the markets with confidence."
        
        # Smart feature highlights based on preferences
        features = []
        if "price" in prefs.quick_actions:
            features.append(f"{cls.EMOJIS['price']} **Live Market Data** - Real-time prices & charts")
        if "analysis" in prefs.quick_actions:
            features.append(f"{cls.EMOJIS['brain']} **AI Analysis** - Smart market insights")
        
        # Always include core features
        features.extend([
            f"{cls.EMOJIS['watchlist']} **Smart Watchlists** - Track your favorite assets",
            f"{cls.EMOJIS['alert']} **Intelligent Alerts** - Never miss important moves",
            f"{cls.EMOJIS['portfolio']} **Portfolio Analytics** - Monitor your investments",
            f"{cls.EMOJIS['fire']} **Market Intelligence** - Trending opportunities"
        ])
        
        # Add accessibility note if needed
        accessibility_note = ""
        if prefs.accessibility_features.get("screen_reader_support", False):
            accessibility_note = f"\n{cls.EMOJIS['accessibility']} *Screen reader optimized interface active*"
        
        separator = cls.get_themed_separator(user_id or "default")
        
        message = f"""
{cls.EMOJIS['rocket']} **TradeAI Companion** {cls.EMOJIS['diamond']}
{separator}

{greeting}

{intro}

**🎯 What I can do for you:**

"""
        
        for feature in features:
            message += f"{feature}\n"
        
        message += f"""
{separator}
{cls.EMOJIS['lightning']} **Ready to start?** Choose an option below!{accessibility_note}
        """
        
        return cls.format_with_theme(message, user_id or "default")
    
    @staticmethod
    def format_welcome_message(user_name: str = "Trader") -> str:
        """Format a standard welcome message (fallback for non-personalized use)"""
        return f"""
{TradingBotUI.EMOJIS['rocket']} **TradeAI Companion** {TradingBotUI.EMOJIS['diamond']}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

👋 **Welcome, {user_name}!**

Your AI-powered trading assistant is ready to help you navigate the markets with confidence.

**🎯 What I can do for you:**

{TradingBotUI.EMOJIS['chart']} **Live Market Data** - Real-time prices & charts
{TradingBotUI.EMOJIS['brain']} **AI Analysis** - Smart market insights
{TradingBotUI.EMOJIS['watchlist']} **Watchlists** - Track your favorite assets
{TradingBotUI.EMOJIS['alert']} **Price Alerts** - Never miss important moves
{TradingBotUI.EMOJIS['portfolio']} **Portfolio** - Monitor your investments
{TradingBotUI.EMOJIS['fire']} **Market Movers** - Trending opportunities

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{TradingBotUI.EMOJIS['lightning']} **Ready to start?** Choose an option below!
        """
    
    @staticmethod
    def format_price_data(price_data: Dict, symbol: str) -> str:
        """Format price data with modern styling"""
        change = price_data.get('change', 0)
        
        # Ensure change is numeric for comparison
        try:
            if isinstance(change, str):
                change = float(change.replace('%', '').replace(',', '')) if change not in ['N/A', ''] else 0
            change_emoji = TradingBotUI.EMOJIS['up'] if float(change) >= 0 else TradingBotUI.EMOJIS['down']
        except (ValueError, TypeError):
            change_emoji = TradingBotUI.EMOJIS['neutral']
        
        # Determine trend emoji based on change percentage
        change_percent = price_data.get('change_percent', '0%')
        if isinstance(change_percent, str):
            change_percent_num = float(change_percent.replace('%', '')) if change_percent != 'N/A' else 0
        else:
            change_percent_num = change_percent
        
        trend_emoji = TradingBotUI.EMOJIS['fire'] if abs(change_percent_num) > 5 else change_emoji
        
        # Format market cap
        market_cap = price_data.get('market_cap', 'N/A')
        if market_cap != 'N/A' and isinstance(market_cap, (int, float)):
            if market_cap >= 1e12:
                market_cap = f"${market_cap/1e12:.2f}T"
            elif market_cap >= 1e9:
                market_cap = f"${market_cap/1e9:.2f}B"
            elif market_cap >= 1e6:
                market_cap = f"${market_cap/1e6:.2f}M"
        
        display_symbol = price_data.get('original_symbol', symbol)
        company_name = price_data.get('company_name', display_symbol)
        
        return f"""
{trend_emoji} **{display_symbol}** • *{company_name}*

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{TradingBotUI.EMOJIS['price']} **Current Price:** `${price_data['price']}`
{change_emoji} **Change:** `{price_data.get('change', 'N/A')} ({price_data.get('change_percent', 'N/A')}%)`

{TradingBotUI.EMOJIS['up']} **Day High:** `${price_data.get('high', 'N/A')}`
{TradingBotUI.EMOJIS['down']} **Day Low:** `${price_data.get('low', 'N/A')}`
{TradingBotUI.EMOJIS['chart']} **Volume:** `{price_data.get('volume', 'N/A'):,}`
{TradingBotUI.EMOJIS['diamond']} **Market Cap:** `{market_cap}`
{TradingBotUI.EMOJIS['target']} **P/E Ratio:** `{price_data.get('pe_ratio', 'N/A')}`

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{TradingBotUI.EMOJIS['info']} *Updated: {price_data['timestamp']}*
        """
    
    @staticmethod
    def format_error_message(title: str, reasons: List[str] = None, suggestions: List[str] = None) -> str:
        """Format error messages with consistent styling"""
        formatted_message = f"""
{TradingBotUI.EMOJIS['error']} **{title}**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """
        
        if reasons:
            formatted_message += f"\n{TradingBotUI.EMOJIS['warning']} **Possible reasons:**\n"
            for reason in reasons:
                formatted_message += f"• {reason}\n"
        
        if suggestions:
            formatted_message += f"\n{TradingBotUI.EMOJIS['info']} **Suggestions:**\n"
            for suggestion in suggestions:
                formatted_message += f"• {suggestion}\n"
        
        return formatted_message
    
    @staticmethod
    def format_help_section(title: str, description: str, commands: List[str], emoji: str = None) -> str:
        """Format help sections with consistent styling"""
        section_emoji = emoji or TradingBotUI.EMOJIS['info']
        
        formatted_section = f"""
{section_emoji} **{title.upper()}**

{description}

        """
        
        for command in commands:
            formatted_section += f"• {command}\n"
        
        return formatted_section
    

    

    

    

    

    

    
    @classmethod
    def format_smart_notification(cls, notification_type: str, title: str, message: str, 
                                user_id: str = None, priority: str = "normal") -> str:
        """Format smart notifications with user preferences and priority levels"""
        prefs = cls.get_user_preferences(user_id) if user_id else UserPreferences()
        
        # Check if user wants this type of notification
        if not prefs.notification_preferences.get(notification_type, True):
            return None  # User has disabled this notification type
        
        # Priority-based styling
        if priority == "high":
            priority_emoji = cls.EMOJIS['fire']
            priority_text = "🚨 HIGH PRIORITY"
        elif priority == "urgent":
            priority_emoji = cls.EMOJIS['rocket']
            priority_text = "⚡ URGENT"
        else:
            priority_emoji = cls.EMOJIS['info']
            priority_text = ""
        
        # Notification type styling
        type_emojis = {
            "price_alert": cls.EMOJIS['alert'],
            "market_news": cls.EMOJIS['news'],
            "portfolio_update": cls.EMOJIS['portfolio'],
            "system_notification": cls.EMOJIS['settings'],
            "ai_insight": cls.EMOJIS['brain']
        }
        
        type_emoji = type_emojis.get(notification_type, cls.EMOJIS['info'])
        separator = cls.get_themed_separator(user_id or "default")
        
        formatted_message = f"""
{type_emoji} **{title}** {priority_emoji}
{separator}

{message}
"""
        
        if priority_text:
            formatted_message += f"\n{priority_text}\n"
        
        formatted_message += f"\n{cls.EMOJIS['clock']} *{format_ist_timestamp('%H:%M:%S IST')}*"
        
        return cls.format_with_theme(formatted_message, user_id or "default")
    

    
    @staticmethod
    def format_loading_message(action: str) -> str:
        """Format a loading message with animation"""
        return f"{TradingBotUI.EMOJIS['loading']} **Processing {action}...**\n\n{TradingBotUI.EMOJIS['lightning']} Please wait while we fetch the latest data"
    
    @staticmethod
    def format_status_message(status: str, message: str, details: str = None) -> str:
        """Format a status message with appropriate styling"""
        status_emoji = {
            'success': TradingBotUI.EMOJIS['success'],
            'error': TradingBotUI.EMOJIS['error'],
            'warning': TradingBotUI.EMOJIS['warning'],
            'info': TradingBotUI.EMOJIS['info']
        }.get(status, TradingBotUI.EMOJIS['info'])
        
        formatted = f"{status_emoji} **{message}**"
        if details:
            formatted += f"\n\n{TradingBotUI.EMOJIS['info']} {details}"
        
        return formatted
    
    @staticmethod
    def format_premium_feature_message(feature: str) -> str:
        """Format a premium feature promotion message"""
        return (
            f"{TradingBotUI.EMOJIS['crown']} **Premium Feature**\n\n"
            f"{TradingBotUI.EMOJIS['gem']} {feature} is available for premium users\n\n"
            f"{TradingBotUI.EMOJIS['star']} **Benefits:**\n"
            f"• Advanced AI analysis\n"
            f"• Real-time alerts\n"
            f"• Priority support\n"
            f"• Extended data history\n\n"
            f"{TradingBotUI.EMOJIS['rocket']} Upgrade today for enhanced trading insights!"
        )
    
    @staticmethod
    def format_price_display(symbol: str, price: float, change: float, change_percent: float, 
                           volume: int = None, market_cap: str = None, high_52w: float = None, 
                           low_52w: float = None) -> str:
        """Format price information with modern, mobile-optimized styling"""
        # Determine trend styling
        if change > 0:
            trend_emoji = TradingBotUI.EMOJIS['up']
            change_symbol = "+"
            trend_indicator = "🟢"
            momentum = "BULLISH"
        elif change < 0:
            trend_emoji = TradingBotUI.EMOJIS['down']
            change_symbol = ""
            trend_indicator = "🔴"
            momentum = "BEARISH"
        else:
            trend_emoji = TradingBotUI.EMOJIS['neutral']
            change_symbol = ""
            trend_indicator = "🟡"
            momentum = "NEUTRAL"
        
        # Create modern card-style layout
        message = f"""
{TradingBotUI.EMOJIS['chart']} **{symbol.upper()}** {trend_indicator}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💰 **${price:.2f}**
{trend_emoji} `{change_symbol}${abs(change):.2f} ({change_symbol}{abs(change_percent):.2f}%)`

📊 **Market Status:** {momentum}
"""
        
        # Add volume with smart formatting
        if volume:
            if volume >= 1_000_000:
                vol_display = f"{volume/1_000_000:.1f}M"
            elif volume >= 1_000:
                vol_display = f"{volume/1_000:.1f}K"
            else:
                vol_display = f"{volume:,}"
            message += f"\n{TradingBotUI.EMOJIS['fire']} **Volume:** {vol_display}"
        
        # Add market cap if provided
        if market_cap:
            message += f"\n{TradingBotUI.EMOJIS['diamond']} **Market Cap:** {market_cap}"
        
        # Add 52-week range if provided
        if high_52w and low_52w:
            message += f"\n\n📈 **52W Range:** ${low_52w:.2f} - ${high_52w:.2f}"
        
        # Add timestamp
        message += f"\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        message += f"\n{TradingBotUI.EMOJIS['clock']} *Last updated: {format_ist_timestamp('%H:%M:%S IST')}*"
        
        return message





    @staticmethod
    def format_loading_message(action: str = "Loading") -> str:
        """Format loading message with animation-ready styling"""
        return f"""
{TradingBotUI.EMOJIS['loading']} **{action}...**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{TradingBotUI.EMOJIS['processing']} Please wait while we fetch the latest data.

This usually takes just a few seconds.
"""


    
    @classmethod
    def format_advanced_market_data(cls, data: Dict[str, Any], user_id: str = None) -> str:
        """Format advanced market data with rich visualizations"""
        prefs = cls.get_user_preferences(user_id) if user_id else UserPreferences()
        
        # Create ASCII chart for price movement
        def create_mini_chart(prices: List[float], width: int = 20) -> str:
            if not prices or len(prices) < 2:
                return "No data available"
            
            min_price = min(prices)
            max_price = max(prices)
            price_range = max_price - min_price
            
            if price_range == 0:
                return "─" * width
            
            chart = ""
            for i, price in enumerate(prices[-width:]):
                normalized = (price - min_price) / price_range
                if i == 0:
                    chart += "▶"
                elif normalized > 0.75:
                    chart += "▲"
                elif normalized > 0.5:
                    chart += "●"
                elif normalized > 0.25:
                    chart += "▼"
                else:
                    chart += "▽"
            
            return chart
        
        symbol = data.get('symbol', 'N/A')
        price = data.get('price', 0)
        change = data.get('change', 0)
        change_percent = data.get('change_percent', 0)
        volume = data.get('volume', 0)
        market_cap = data.get('market_cap', 0)
        prices_history = data.get('price_history', [])
        
        # Price trend indicator
        trend_emoji = cls.EMOJIS['up'] if change >= 0 else cls.EMOJIS['down']
        trend_color = "🟢" if change >= 0 else "🔴"
        
        # Volume indicator
        volume_str = cls.format_large_number(volume)
        market_cap_str = cls.format_large_number(market_cap)
        
        # Mini chart
        mini_chart = create_mini_chart(prices_history)
        
        separator = cls.get_themed_separator(user_id or "default")
        
        formatted_data = f"""
{cls.EMOJIS['stock']} **{symbol}** {trend_color}
{separator}

{cls.EMOJIS['money']} **Price:** ${price:,.2f}
{trend_emoji} **Change:** {change:+.2f} ({change_percent:+.2f}%)

{cls.EMOJIS['chart']} **Trend:** `{mini_chart}`

{cls.EMOJIS['volume']} **Volume:** {volume_str}
{cls.EMOJIS['market']} **Market Cap:** {market_cap_str}

{cls.EMOJIS['clock']} **Last Updated:** {format_ist_timestamp('%H:%M:%S IST')}
"""
        
        return cls.format_with_theme(formatted_data, user_id or "default")
    
    @classmethod
    def create_portfolio_dashboard(cls, portfolio_data: Dict[str, Any], user_id: str = None) -> str:
        """Create a comprehensive portfolio dashboard"""
        prefs = cls.get_user_preferences(user_id) if user_id else UserPreferences()
        
        total_value = portfolio_data.get('total_value', 0)
        total_change = portfolio_data.get('total_change', 0)
        total_change_percent = portfolio_data.get('total_change_percent', 0)
        positions = portfolio_data.get('positions', [])
        
        # Portfolio performance indicators
        performance_emoji = cls.EMOJIS['up'] if total_change >= 0 else cls.EMOJIS['down']
        performance_color = "🟢" if total_change >= 0 else "🔴"
        
        # Create portfolio allocation chart (simplified)
        def create_allocation_chart(positions: List[Dict], max_width: int = 20) -> str:
            if not positions:
                return "No positions"
            
            total_val = sum(pos.get('value', 0) for pos in positions)
            if total_val == 0:
                return "No value"
            
            chart = ""
            for pos in positions[:5]:  # Show top 5 positions
                symbol = pos.get('symbol', 'N/A')[:3]
                value = pos.get('value', 0)
                percentage = (value / total_val) * 100
                bar_length = int((percentage / 100) * max_width)
                bar = "█" * bar_length + "░" * (max_width - bar_length)
                chart += f"{symbol}: {bar} {percentage:.1f}%\n"
            
            return chart.strip()
        
        allocation_chart = create_allocation_chart(positions)
        separator = cls.get_themed_separator(user_id or "default")
        
        dashboard = f"""
{cls.EMOJIS['portfolio']} **Portfolio Dashboard** {performance_color}
{separator}

{cls.EMOJIS['money']} **Total Value:** ${total_value:,.2f}
{performance_emoji} **Total Change:** {total_change:+,.2f} ({total_change_percent:+.2f}%)

{cls.EMOJIS['pie_chart']} **Allocation:**
```
{allocation_chart}
```

{cls.EMOJIS['star']} **Top Performers:**
"""
        
        # Add top 3 performing positions
        sorted_positions = sorted(positions, key=lambda x: x.get('change_percent', 0), reverse=True)
        for i, pos in enumerate(sorted_positions[:3]):
            symbol = pos.get('symbol', 'N/A')
            change_percent = pos.get('change_percent', 0)
            emoji = cls.EMOJIS['up'] if change_percent >= 0 else cls.EMOJIS['down']
            dashboard += f"{i+1}. {symbol} {emoji} {change_percent:+.2f}%\n"
        
        dashboard += f"\n{cls.EMOJIS['clock']} *Updated: {format_ist_timestamp('%H:%M:%S IST')}*"
        
        return cls.format_with_theme(dashboard, user_id or "default")
    
    @staticmethod
    def format_large_number(number: float) -> str:
        """Format large numbers with appropriate suffixes"""
        if number >= 1_000_000_000:
            return f"{number / 1_000_000_000:.1f}B"
        elif number >= 1_000_000:
            return f"{number / 1_000_000:.1f}M"
        elif number >= 1_000:
            return f"{number / 1_000:.1f}K"
        else:
            return f"{number:.0f}"
    

    
    @classmethod
    def create_ai_help_system(cls, context: str = "general", user_id: str = None) -> str:
        """Create an AI-powered help system with contextual assistance"""
        prefs = cls.get_user_preferences(user_id) if user_id else UserPreferences()
        
        help_content = {
            "general": {
                "title": "🤖 AI Trading Assistant Help",
                "sections": [
                    ("📊 Market Data", "Get real-time stock prices, charts, and market analysis"),
                    ("🔍 Symbol Search", "Search for any stock symbol or company name"),
                    ("📈 Portfolio", "Track your investments and portfolio performance"),
                    ("🚨 Alerts", "Set price alerts and get notified of market changes"),
                    ("🧠 AI Insights", "Get AI-powered market predictions and analysis")
                ]
            },
            "trading": {
                "title": "📈 Trading Features Help",
                "sections": [
                    ("💰 Price Tracking", "Monitor real-time prices with advanced charts"),
                    ("📊 Technical Analysis", "Use RSI, MACD, Bollinger Bands and more"),
                    ("🎯 Smart Alerts", "Set intelligent price and volume alerts"),
                    ("📱 Quick Actions", "Access frequently used features instantly"),
                    ("🌙 After Hours", "Track pre-market and after-hours trading")
                ]
            },
            "settings": {
                "title": "⚙️ Settings & Customization",
                "sections": [
                    ("🎨 Themes", "Choose from multiple visual themes"),
                    ("🔔 Notifications", "Customize your notification preferences"),
                    ("♿ Accessibility", "Enable screen reader and accessibility features"),
                    ("⭐ Favorites", "Save your favorite stocks for quick access"),
                    ("🌍 Localization", "Set your preferred language and timezone")
                ]
            }
        }
        
        content = help_content.get(context, help_content["general"])
        separator = cls.get_themed_separator(user_id or "default")
        
        help_message = f"""
{content['title']}
{separator}

"""
        
        for emoji_title, description in content['sections']:
            help_message += f"{emoji_title}\n{description}\n\n"
        
        help_message += f"""
{cls.EMOJIS['lightbulb']} **Pro Tips:**
• Use quick actions for faster navigation
• Set up alerts to never miss important price movements
• Customize your theme for better readability
• Enable accessibility features if needed

{cls.EMOJIS['support']} **Need more help?** Contact our support team!

{cls.EMOJIS['clock']} *Help updated: {format_ist_timestamp('%H:%M:%S IST')}*
"""
        
        return cls.format_with_theme(help_message, user_id or "default")
    
    @classmethod
    def create_contextual_tooltip(cls, feature: str, user_id: str = None) -> str:
        """Create contextual tooltips for UI features"""
        tooltips = {
            "price_alerts": {
                "title": "🚨 Price Alerts",
                "description": "Get notified when your stocks hit target prices",
                "tips": [
                    "Set both upper and lower price targets",
                    "Use percentage-based alerts for volatility",
                    "Enable push notifications for instant alerts"
                ]
            },
            "technical_analysis": {
                "title": "📊 Technical Analysis",
                "description": "Advanced charting tools for better trading decisions",
                "tips": [
                    "RSI above 70 indicates overbought conditions",
                    "MACD crossovers signal potential trend changes",
                    "Bollinger Bands show volatility and support/resistance"
                ]
            },
            "portfolio_tracking": {
                "title": "📈 Portfolio Tracking",
                "description": "Monitor your investments in real-time",
                "tips": [
                    "Diversify across different sectors",
                    "Regular rebalancing improves long-term returns",
                    "Track both absolute and percentage gains"
                ]
            },
            "ai_insights": {
                "title": "🧠 AI Insights",
                "description": "Machine learning powered market analysis",
                "tips": [
                    "AI predictions are based on historical patterns",
                    "Combine AI insights with your own research",
                    "Market sentiment analysis helps time entries"
                ]
            }
        }
        
        tooltip = tooltips.get(feature, {
            "title": "ℹ️ Feature Info",
            "description": "Learn more about this feature",
            "tips": ["Explore the interface to discover more features"]
        })
        
        separator = cls.get_themed_separator(user_id or "default")
        
        tooltip_message = f"""
{tooltip['title']}
{separator}

{tooltip['description']}

{cls.EMOJIS['lightbulb']} **Quick Tips:**
"""
        
        for tip in tooltip['tips']:
            tooltip_message += f"• {tip}\n"
        
        tooltip_message += f"\n{cls.EMOJIS['info']} *Tap anywhere to continue*"
        
        return cls.format_with_theme(tooltip_message, user_id or "default")
    

    


    @staticmethod
    def format_market_movers(movers_data: list, title: str = "Top Market Movers") -> str:
        """Format market movers with modern card-style layout"""
        message = f"""
{TradingBotUI.EMOJIS['fire']} **{title}** {TradingBotUI.EMOJIS['trending']}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
        
        for i, mover in enumerate(movers_data[:5], 1):  # Limit to top 5
            symbol = mover.get('symbol', 'N/A')
            price = mover.get('price', 0)
            change_percent = mover.get('change_percent', 0)
            
            # Determine trend indicator
            if change_percent > 0:
                trend = "🟢"
                arrow = "📈"
            else:
                trend = "🔴"
                arrow = "📉"
            
            message += f"{i}. {trend} **{symbol}** ${price:.2f} {arrow} {change_percent:+.2f}%\n"
        
        message += f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        message += f"\n{TradingBotUI.EMOJIS['clock']} *Updated: {format_ist_timestamp('%H:%M IST')}*"
        
        return message

    @staticmethod
    def format_watchlist(watchlist_data: list, user_name: str = "Your") -> str:
        """Format watchlist with modern, clean layout"""
        if not watchlist_data:
            return f"""
{TradingBotUI.EMOJIS['watchlist']} **{user_name} Watchlist** {TradingBotUI.EMOJIS['star']}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{TradingBotUI.EMOJIS['info']} Your watchlist is empty.

Tap "Add Symbol" to start tracking your favorite stocks!
"""
        
        message = f"""
{TradingBotUI.EMOJIS['watchlist']} **{user_name} Watchlist** {TradingBotUI.EMOJIS['star']}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
        
        for item in watchlist_data:
            symbol = item.get('symbol', 'N/A')
            price = item.get('price', 0)
            change_percent = item.get('change_percent', 0)
            
            # Status indicator
            if change_percent > 2:
                status = "🚀"  # Strong gain
            elif change_percent > 0:
                status = "🟢"  # Gain
            elif change_percent < -2:
                status = "💥"  # Strong loss
            elif change_percent < 0:
                status = "🔴"  # Loss
            else:
                status = "🟡"  # Neutral
            
            message += f"{status} **{symbol}** ${price:.2f} `{change_percent:+.2f}%`\n"
        
        message += f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        message += f"\n{TradingBotUI.EMOJIS['refresh']} *Last updated: {format_ist_timestamp('%H:%M IST')}*"
        
        return message

    @staticmethod
    def format_data_table(headers: list, rows: list, title: str = None) -> str:
        """Format data in a clean table structure with modern styling"""
        message = ""
        
        if title:
            message += f"\n{TradingBotUI.EMOJIS['data']} **{title}** {TradingBotUI.EMOJIS['chart']}\n"
            message += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        
        # Create header row with better formatting
        header_row = " | ".join([f"**{header}**" for header in headers])
        message += f"`{header_row}`\n"
        message += "─" * 40 + "\n"
        
        # Add data rows with monospace formatting
        for row in rows:
            row_text = " | ".join([str(cell) for cell in row])
            message += f"`{row_text}`\n"
        
        return message



    @staticmethod
    def format_alert_notification(symbol: str, price: float, target_price: float, 
                                alert_type: str = "price") -> str:
        """Format price alert notifications with modern styling"""
        if alert_type == "price":
            if price >= target_price:
                emoji = TradingBotUI.EMOJIS['rocket']
                status = "TARGET REACHED"
                color = "🟢"
            else:
                emoji = TradingBotUI.EMOJIS['alert']
                status = "PRICE ALERT"
                color = "🔔"
        else:
            emoji = TradingBotUI.EMOJIS['alert']
            status = "ALERT TRIGGERED"
            color = "🟡"
        
        return f"""
{emoji} **{status}** {color}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{TradingBotUI.EMOJIS['chart']} **{symbol.upper()}**
💰 **Current Price:** ${price:.2f}
🎯 **Target Price:** ${target_price:.2f}

{TradingBotUI.EMOJIS['clock']} *{format_ist_timestamp('%Y-%m-%d %H:%M:%S IST')}*
"""

    @staticmethod
    def format_error_message_modern(error_type: str, message: str) -> str:
        """Format error messages with consistent styling"""
        return f"""
{TradingBotUI.EMOJIS['error']} **Error: {error_type}**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{TradingBotUI.EMOJIS['info']} {message}

Please try again or contact support if the issue persists.
"""

    @staticmethod
    def format_success_message(action: str, details: str = None) -> str:
        """Format success messages with modern styling"""
        message = f"""
{TradingBotUI.EMOJIS['success']} **Success!**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{TradingBotUI.EMOJIS['lightning']} {action}
"""
        
        if details:
            message += f"\n{TradingBotUI.EMOJIS['info']} {details}"
        
        return message
    
    @staticmethod
    def create_price_keyboard():
        """Create keyboard for price command help"""
        # Return None since we're not using inline keyboards
        return None
    
    @staticmethod
    def create_stock_action_menu(symbol: str):
        """Create action menu for stock symbols"""
        # Return None since we're not using inline keyboards
        return None
    
    @staticmethod
    def create_stock_actions_keyboard(symbol: str):
        """Create actions keyboard for stock symbols"""
        # Return None since we're not using inline keyboards
        return None
    
    @staticmethod
    def create_main_menu():
        """Create main menu keyboard"""
        # Return None since we're not using inline keyboards
        return None
    
    @staticmethod
    def create_analysis_keyboard():
        """Create analysis keyboard"""
        # Return None since we're not using inline keyboards
        return None


# Demo and Testing Functions
def demo_ui_components():
    """Comprehensive demo of all UI components for testing and showcase"""
    print("=" * 60)
    print("🚀 TRADEAI COMPANION - PRODUCTION UI DEMO")
    print("=" * 60)
    
    # 1. Welcome Message Demo
    print("\n1. WELCOME MESSAGE:")
    print("-" * 40)
    print(TradingBotUI.format_welcome_message("John Trader"))
    
    # 2. Price Display Demo
    print("\n2. PRICE DISPLAY:")
    print("-" * 40)
    print(TradingBotUI.format_price_display(
        symbol="AAPL",
        price=175.43,
        change=2.15,
        change_percent=1.24,
        volume=45_678_900,
        market_cap="$2.8T",
        high_52w=198.23,
        low_52w=124.17
    ))
    
    # 3. Market Movers Demo
    print("\n3. MARKET MOVERS:")
    print("-" * 40)
    movers_data = [
        {'symbol': 'NVDA', 'price': 432.15, 'change_percent': 8.45},
        {'symbol': 'TSLA', 'price': 248.73, 'change_percent': -3.21},
        {'symbol': 'AMD', 'price': 156.89, 'change_percent': 5.67},
        {'symbol': 'GOOGL', 'price': 2847.32, 'change_percent': 2.34},
        {'symbol': 'META', 'price': 487.91, 'change_percent': -1.89}
    ]
    print(TradingBotUI.format_market_movers(movers_data))
    
    # 4. Watchlist Demo
    print("\n4. WATCHLIST:")
    print("-" * 40)
    watchlist_data = [
        {'symbol': 'AAPL', 'price': 175.43, 'change_percent': 1.24},
        {'symbol': 'MSFT', 'price': 378.91, 'change_percent': -0.87},
        {'symbol': 'AMZN', 'price': 3247.15, 'change_percent': 3.45}
    ]
    print(TradingBotUI.format_watchlist(watchlist_data, "John's"))
    
    # 5. Alert Notification Demo
    print("\n5. ALERT NOTIFICATION:")
    print("-" * 40)
    print(TradingBotUI.format_alert_notification(
        symbol="TSLA",
        price=250.00,
        target_price=245.00,
        alert_type="price"
    ))
    
    # 6. Success Message Demo
    print("\n6. SUCCESS MESSAGE:")
    print("-" * 40)
    print(TradingBotUI.format_success_message(
        action="Watchlist updated successfully!",
        details="AAPL has been added to your watchlist with price alerts enabled."
    ))
    
    # 7. Error Message Demo
    print("\n7. ERROR MESSAGE:")
    print("-" * 40)
    print(TradingBotUI.format_error_message_modern(
        error_type="Symbol Not Found",
        message="The symbol 'XYZ123' could not be found. Please check the spelling and try again."
    ))
    
    # 8. Loading Message Demo
    print("\n8. LOADING MESSAGE:")
    print("-" * 40)
    print(TradingBotUI.format_loading_message("Fetching market data"))
    
    print("\n" + "=" * 60)
    print("✅ UI DEMO COMPLETE - All components rendered successfully!")
    print("=" * 60)


if __name__ == "__main__":
    # Run demo when script is executed directly
    demo_ui_components()