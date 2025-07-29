"""
Chart Service - Generate dynamic charts using Chart-IMG API and matplotlib
Provides visual analysis for trading decisions
"""
import os
import asyncio
import aiohttp
from logger import logger
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    logger.warning("Matplotlib not available - chart generation disabled")
try:
    import matplotlib.dates as mdates
except ImportError:
    mdates = None
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    yf = None
    YFINANCE_AVAILABLE = False
    print("Warning: yfinance not available in chart_service, some features disabled")
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from io import BytesIO
import base64
from config import Config

class ChartService:
    """Service for generating trading charts and visual analysis"""
    
    def __init__(self):
        """Initialize chart service"""
        config = Config()
        self.chart_api_key = config.CHART_IMG_API_KEY
        logger.info("Chart service initialized")
    
    async def generate_price_chart(self, symbol: str, period: str = '1mo', interval: str = '1d') -> Optional[str]:
        """
        Generate a modern TradingView-style price chart using Chart-IMG API only. No fallback.
        """
        try:
            symbol = symbol.upper().strip()
            if not self.chart_api_key:
                logger.error("Chart-IMG API key not available. Cannot generate chart.")
                return None
            try:
                import aiohttp
                logger.info(f"Generating chart for {symbol} via Chart-IMG API (forced) - Period: {period}")
                # Map period parameter to Chart-IMG API interval format
                period_map = {
                    # Standard periods
                    '1d': '1D', '5d': '5D', '1w': '1W', '1wk': '1W', '1mo': '1M', '1M': '1M', 
                    '3mo': '3M', '3M': '3M', '6mo': '6M', '6M': '6M', '1y': '1Y', '1Y': '1Y',
                    '2y': '2Y', '5y': '5Y', '1h': '1H', '1m': '1m',
                    # Common user aliases
                    '6m': '6M',     # 6 months (user shorthand)
                    '1min': '1m',   # 1 minute (alternative format)
                    '5min': '5m',   # 5 minutes
                    '15min': '15m', # 15 minutes
                    '30min': '30m', # 30 minutes
                    '1hr': '1H',    # 1 hour (alternative format)
                    '3m': '3M',     # 3 months (user shorthand)
                    '12m': '1Y'     # 12 months = 1 year
                }
                chartimg_interval = period_map.get(period, '1D')
                if ':' not in symbol:
                    chartimg_symbol = f'NASDAQ:{symbol}'
                else:
                    chartimg_symbol = symbol
                url = 'https://api.chart-img.com/v2/tradingview/advanced-chart'
                headers = {
                    'x-api-key': self.chart_api_key,
                    'content-type': 'application/json'
                }
                payload = {
                    'symbol': chartimg_symbol,
                    'interval': chartimg_interval,
                    'theme': 'dark',
                    'style': 'candle',
                    'width': 800,
                    'height': 600,
                    'studies': [
                        {'name': 'Volume'},
                        {'name': 'Relative Strength Index'},
                        {'name': 'MACD'}
                    ]
                }
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, headers=headers, json=payload, timeout=15) as resp:
                        logger.info(f"Chart-IMG API response status: {resp.status}")
                        if resp.status == 200:
                            img_bytes = await resp.read()
                            import base64
                            chart_b64 = base64.b64encode(img_bytes).decode()
                            logger.info(f"Chart-IMG chart generated successfully for {symbol}")
                            return chart_b64
                        else:
                            error_text = await resp.text()
                            logger.error(f"Chart-IMG API failed for {symbol}: {resp.status} {error_text}")
                            return None
            except Exception as e:
                logger.error(f"Chart-IMG API error for {symbol}: {e}")
                return None
        except Exception as e:
            logger.error(f"Error in generate_price_chart for {symbol}: {e}")
            return None
    
    async def generate_comparison_chart(self, symbols: List[str], period: str = '1mo') -> Optional[str]:
        """
        Generate comparison chart for multiple stocks
        
        Args:
            symbols (List[str]): List of stock symbols
            period (str): Time period
            
        Returns:
            Base64 encoded comparison chart or None if error
        """
        try:
            symbols = [s.upper().strip() for s in symbols[:5]]  # Limit to 5 stocks
            logger.info(f"Generating comparison chart for {symbols}")
            
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(12, 8))
            
            colors = ['#00ff88', '#ff6b6b', '#4ecdc4', '#ffd93d', '#ff8c69']
            
            for i, symbol in enumerate(symbols):
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period=period)
                    
                    if not hist.empty:
                        # Normalize to percentage change
                        normalized = (hist['Close'] / hist['Close'].iloc[0] - 1) * 100
                        ax.plot(hist.index, normalized, color=colors[i % len(colors)], 
                               linewidth=2, label=symbol)
                except Exception as e:
                    logger.warning(f"Failed to fetch data for {symbol}: {e}")
            
            ax.set_title('Stock Performance Comparison', fontsize=16, color='white', pad=20)
            ax.set_ylabel('Change (%)', fontsize=12)
            ax.set_xlabel('Date', fontsize=12)
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='white', linestyle='--', alpha=0.5)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight',
                       facecolor='#1a1a1a', edgecolor='none')
            buffer.seek(0)
            
            chart_b64 = base64.b64encode(buffer.read()).decode()
            plt.close()
            
            logger.info("Comparison chart generated successfully")
            return chart_b64
            
        except Exception as e:
            logger.error(f"Error generating comparison chart: {e}")
            return None
    
    async def generate_sector_chart(self, sector_data: Dict) -> Optional[str]:
        """
        Generate sector performance chart
        
        Args:
            sector_data (Dict): Sector performance data
            
        Returns:
            Base64 encoded sector chart or None if error
        """
        try:
            if not sector_data.get('sectors'):
                return None
            
            logger.info("Generating sector performance chart")
            
            sectors = sector_data['sectors']
            names = [s['sector'] for s in sectors]
            changes = [s['change_percent'] for s in sectors]
            
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Color bars based on performance
            colors = ['#00ff88' if change >= 0 else '#ff6b6b' for change in changes]
            
            bars = ax.barh(names, changes, color=colors, alpha=0.8)
            
            # Add value labels on bars
            for i, (bar, change) in enumerate(zip(bars, changes)):
                ax.text(bar.get_width() + (0.1 if change >= 0 else -0.1), bar.get_y() + bar.get_height()/2,
                       f'{change:+.2f}%', ha='left' if change >= 0 else 'right', va='center',
                       fontweight='bold', color='white')
            
            ax.set_title('Sector Performance Today', fontsize=16, color='white', pad=20)
            ax.set_xlabel('Change (%)', fontsize=12)
            ax.grid(True, alpha=0.3, axis='x')
            ax.axvline(x=0, color='white', linestyle='-', alpha=0.5)
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight',
                       facecolor='#1a1a1a', edgecolor='none')
            buffer.seek(0)
            
            chart_b64 = base64.b64encode(buffer.read()).decode()
            plt.close()
            
            logger.info("Sector chart generated successfully")
            return chart_b64
            
        except Exception as e:
            logger.error(f"Error generating sector chart: {e}")
            return None
    
    async def generate_portfolio_chart(self, portfolio_data: Dict) -> Optional[str]:
        """
        Generate portfolio allocation chart
        
        Args:
            portfolio_data (Dict): Portfolio data
            
        Returns:
            Base64 encoded portfolio chart or None if error
        """
        try:
            if not portfolio_data.get('positions'):
                return None
            
            logger.info("Generating portfolio allocation chart")
            
            positions = portfolio_data['positions']
            symbols = [p['symbol'] for p in positions]
            values = [p['market_value'] for p in positions]
            
            plt.style.use('dark_background')
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
            
            # Pie chart
            colors = plt.cm.Set3(np.linspace(0, 1, len(symbols)))
            wedges, texts, autotexts = ax1.pie(values, labels=symbols, autopct='%1.1f%%',
                                              colors=colors, startangle=90)
            ax1.set_title('Portfolio Allocation', fontsize=14, color='white')
            
            # Bar chart with P&L
            pnl = [p['unrealized_pl'] for p in positions]
            bar_colors = ['#00ff88' if p >= 0 else '#ff6b6b' for p in pnl]
            
            bars = ax2.bar(symbols, pnl, color=bar_colors, alpha=0.8)
            ax2.set_title('Unrealized P&L by Position', fontsize=14, color='white')
            ax2.set_ylabel('P&L ($)', fontsize=12)
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.axhline(y=0, color='white', linestyle='-', alpha=0.5)
            
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight',
                       facecolor='#1a1a1a', edgecolor='none')
            buffer.seek(0)
            
            chart_b64 = base64.b64encode(buffer.read()).decode()
            plt.close()
            
            logger.info("Portfolio chart generated successfully")
            return chart_b64
            
        except Exception as e:
            logger.error(f"Error generating portfolio chart: {e}")
            return None
    
    async def generate_chart_via_api(self, chart_config: Dict) -> Optional[str]:
        """
        Generate chart using Chart-IMG API (if available)
        
        Args:
            chart_config (Dict): Chart configuration
            
        Returns:
            Chart URL or None if error
        """
        try:
            if not self.chart_api_key:
                logger.warning("Chart-IMG API key not available")
                return None
            
            logger.info("Generating chart via Chart-IMG API")
            
            # Chart-IMG API integration would go here
            # This is a placeholder for the actual API integration
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating chart via API: {e}")
            return None