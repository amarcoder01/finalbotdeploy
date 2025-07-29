"""
Market Data Service - Real-time market data using multiple sources
Provides comprehensive market information for trading decisions across global markets
"""
import os
import asyncio
import aiohttp
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    yf = None
    YFINANCE_AVAILABLE = False
    print("Warning: yfinance not available, Yahoo Finance data disabled")
try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    np = None
    PANDAS_AVAILABLE = False
    print("Warning: Pandas/NumPy not available, using limited functionality")
from datetime import datetime, timedelta
from timezone_utils import format_ist_timestamp
from typing import Dict, List, Optional, Tuple, Any
try:
    from alpaca.data import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestQuoteRequest
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetAssetsRequest
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
from logger import logger
from config import Config

class MarketDataService:
    """Service for fetching real-time market data and analytics from multiple sources"""
    
    def __init__(self):
        """Initialize market data clients"""
        self.alpaca_data_client = None
        self.alpaca_trading_client = None
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
        
        # Initialize Alpaca client if available and keys are provided
        if ALPACA_AVAILABLE and Config.ALPACA_API_KEY and Config.ALPACA_API_SECRET:
            try:
                self.alpaca_data_client = StockHistoricalDataClient(
                    api_key=Config.ALPACA_API_KEY,
                    secret_key=Config.ALPACA_API_SECRET
                )
                self.alpaca_trading_client = TradingClient(
                    api_key=Config.ALPACA_API_KEY,
                    secret_key=Config.ALPACA_API_SECRET,
                    paper=True
                )
                logger.info("Alpaca API clients initialized successfully")
            except Exception as e:
                logger.warning(f"Alpaca API initialization failed: {e}")
        elif not ALPACA_AVAILABLE:
            logger.info("Alpaca API not available - using alternative sources")
                
        logger.info("Multi-source market data service initialized")
    
    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol for any US stock - no hardcoded mappings"""
        symbol = symbol.upper().strip()
        # Remove any common prefixes/suffixes that might cause issues
        symbol = symbol.replace('.US', '').replace('.O', '').replace('.Q', '')
        # Handle common variations
        if symbol.endswith('.PINK'):
            symbol = symbol.replace('.PINK', '')
        if symbol.endswith('.OTC'):
            symbol = symbol.replace('.OTC', '')
        # Return clean symbol - let the data sources handle the rest
        return symbol

    async def _try_alpaca_api(self, symbol: str) -> Optional[Dict]:
        """Try to get data from Alpaca API"""
        if not self.alpaca_data_client:
            return None
        
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=[symbol])
            quotes = self.alpaca_data_client.get_stock_latest_quote(request)
            if quotes and hasattr(quotes, symbol) and getattr(quotes, symbol):
                quote = getattr(quotes, symbol)
                return {
                    'symbol': symbol,
                    'price': float(quote.ask_price + quote.bid_price) / 2,
                    'bid': float(quote.bid_price),
                    'ask': float(quote.ask_price),
                    'timestamp': quote.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'source': 'Alpaca'
                }
        except Exception as e:
            logger.warning(f"Alpaca API failed for {symbol}: {str(e)}")
        return None

    async def _try_google_finance(self, symbol: str) -> Optional[Dict]:
        """Try to get data from Google Finance via web scraping"""
        try:
            # Use a simple web scraping approach for Google Finance
            url = f"https://www.google.com/finance/quote/{symbol}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=5) as response:  # Reduced timeout to 5 seconds
                    if response.status == 200:
                        html = await response.text()
                        # Simple regex extraction (in production, use proper HTML parsing)
                        import re
                        
                        # Extract price
                        price_match = re.search(r'"price":\s*"([^"]+)"', html)
                        if price_match:
                            price = float(price_match.group(1).replace(',', ''))
                            
                            # Extract company name
                            name_match = re.search(r'"name":\s*"([^"]+)"', html)
                            company_name = name_match.group(1) if name_match else symbol
                            
                            return {
                                'symbol': symbol,
                                'price': price,
                                'company_name': company_name,
                                'timestamp': format_ist_timestamp('%Y-%m-%d %H:%M:%S IST'),
                                'source': 'Google Finance'
                            }
        except Exception as e:
            logger.warning(f"Google Finance failed for {symbol}: {str(e)}")
        return None

    async def _try_yahoo_finance_global(self, symbol: str) -> Optional[Dict]:
        """Try to get data from Yahoo Finance for US stocks only"""
        if not YFINANCE_AVAILABLE or yf is None:
            return None
            
        # Only try US stock format (no suffixes)
        try:
            logger.info(f"Trying Yahoo Finance US for {symbol}")
            
            # Add delay to prevent rate limiting
            await asyncio.sleep(0.5)
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            # Add timeout to make it fail faster
            hist = await asyncio.wait_for(
                asyncio.to_thread(ticker.history, period='1d', interval='1m'),
                timeout=2.0  # 2 second timeout for US stocks
            )
            
            # Get daily data for volume (1m data often has zero volume for latest entries)
            hist_daily = await asyncio.wait_for(
                asyncio.to_thread(ticker.history, period='1d', interval='1d'),
                timeout=2.0
            )
            
            if not hist.empty and len(hist) > 0:
                current_price = hist['Close'].iloc[-1]
                if current_price > 0:  # Valid price
                    # Get previous close price (yesterday's close, not today's open)
                    previous_close = hist['Close'].iloc[-2] if len(hist) > 1 else hist['Open'].iloc[0]
                    change = current_price - previous_close
                    change_percent = (change / previous_close) * 100 if previous_close > 0 else 0
                    
                    # Get volume from daily data if available, otherwise try minute data
                    volume = 0
                    if not hist_daily.empty and len(hist_daily) > 0:
                        volume = int(hist_daily['Volume'].iloc[-1]) if not pd.isna(hist_daily['Volume'].iloc[-1]) else 0
                    elif not hist.empty and len(hist) > 0:
                        volume = int(hist['Volume'].iloc[-1]) if not pd.isna(hist['Volume'].iloc[-1]) else 0
                    
                    return {
                        'symbol': symbol,
                        'price': round(float(current_price), 2),
                        'change': round(float(change), 2),
                        'change_percent': round(float(change_percent), 3),
                        'volume': volume,
                        'high': round(float(hist['High'].max()), 2),
                        'low': round(float(hist['Low'].min()), 2),
                        'open': round(float(hist['Open'].iloc[0]), 2),
                        'company_name': info.get('longName', symbol),
                        'market_cap': info.get('marketCap', 'N/A'),
                        'pe_ratio': info.get('trailingPE', 'N/A'),
                        'timestamp': format_ist_timestamp('%Y-%m-%d %H:%M:%S IST'),
                        'source': 'Yahoo Finance US',
                        'resolved_symbol': symbol
                    }
        except Exception as e:
            logger.debug(f"Yahoo Finance US failed for {symbol}: {str(e)}")
        
        logger.info(f"Yahoo Finance US failed for {symbol}")
        return None

    async def _try_yfinance(self, symbol: str) -> Optional[Dict]:
        """Try to get data from Yahoo Finance"""
        if not YFINANCE_AVAILABLE or yf is None:
            return None
            
        try:
            # Add delay to prevent rate limiting
            await asyncio.sleep(0.5)
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            # Add timeout to make it fail faster
            hist = await asyncio.wait_for(
                asyncio.to_thread(ticker.history, period='1d', interval='1m'),
                timeout=2.0  # 2 second timeout for faster failure
            )
            
            # Get daily data for volume (1m data often has zero volume for latest entries)
            hist_daily = await asyncio.wait_for(
                asyncio.to_thread(ticker.history, period='1d', interval='1d'),
                timeout=2.0
            )
            
            if not hist.empty and len(hist) > 0:
                current_price = hist['Close'].iloc[-1]
                if current_price > 0:  # Valid price
                    # Get previous close price (yesterday's close, not today's open)
                    previous_close = hist['Close'].iloc[-2] if len(hist) > 1 else hist['Open'].iloc[0]
                    change = current_price - previous_close
                    change_percent = (change / previous_close) * 100 if previous_close > 0 else 0
                    
                    # Get volume from daily data if available, otherwise try minute data
                    volume = 0
                    if not hist_daily.empty and len(hist_daily) > 0:
                        volume = int(hist_daily['Volume'].iloc[-1]) if not pd.isna(hist_daily['Volume'].iloc[-1]) else 0
                    elif not hist.empty and len(hist) > 0:
                        volume = int(hist['Volume'].iloc[-1]) if not pd.isna(hist['Volume'].iloc[-1]) else 0
                    
                    return {
                        'symbol': symbol,
                        'price': round(float(current_price), 2),
                        'change': round(float(change), 2),
                        'change_percent': round(float(change_percent), 3),
                        'volume': volume,
                        'high': round(float(hist['High'].max()), 2),
                        'low': round(float(hist['Low'].min()), 2),
                        'open': round(float(hist['Open'].iloc[0]), 2),
                        'company_name': info.get('longName', symbol),
                        'market_cap': info.get('marketCap', 'N/A'),
                        'pe_ratio': info.get('trailingPE', 'N/A'),
                        'timestamp': format_ist_timestamp('%Y-%m-%d %H:%M:%S IST'),
                        'source': 'Yahoo Finance'
                    }
        except Exception as e:
            logger.warning(f"Yahoo Finance failed for {symbol}: {str(e)}")
        return None

    async def _try_alpha_vantage(self, symbol: str) -> Optional[Dict]:
        """Try to get data from Alpha Vantage API"""
        if not self.alpha_vantage_key:
            return None
        
        try:
            url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={self.alpha_vantage_key}"
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'Global Quote' in data and data['Global Quote']:
                            quote = data['Global Quote']
                            return {
                                'symbol': symbol,
                                'price': float(quote.get('05. price', 0)),
                                'change': float(quote.get('09. change', 0)),
                                'change_percent': float(quote.get('10. change percent', '0%').replace('%', '')),
                                'volume': int(quote.get('06. volume', 0)),
                                'high': float(quote.get('03. high', 0)),
                                'low': float(quote.get('04. low', 0)),
                                'open': float(quote.get('02. open', 0)),
                                'company_name': symbol,
                                'timestamp': format_ist_timestamp('%Y-%m-%d %H:%M:%S IST'),
                                'source': 'Alpha Vantage'
                            }
        except Exception as e:
            logger.warning(f"Alpha Vantage failed for {symbol}: {str(e)}")
        return None

    async def _try_alternative_symbols(self, symbol: str) -> Optional[Dict]:
        """Try alternative symbol formats for global markets"""
        # This method is now replaced by _try_yahoo_finance_global
        return await self._try_yahoo_finance_global(symbol)
    
    async def _try_openai_web_search(self, symbol: str, user_id: int) -> Optional[Dict]:
        """Try to get data using OpenAI's web search capability"""
        try:
            from openai_service import OpenAIService
            openai_service = OpenAIService()
            result = await openai_service.search_stock_price(symbol, user_id)
            
            if result:
                # Parse the structured response from OpenAI
                lines = result.strip().split('\n')
                data = {}
                for line in lines:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        data[key.strip()] = value.strip()
                
                if 'PRICE' in data:
                    # Ensure all numeric fields are properly converted
                    try:
                        price = float(data['PRICE'].replace('$', '').replace(',', '').split()[0])
                        change = float(data.get('CHANGE', '0').replace(',', '')) if data.get('CHANGE', 'N/A') != 'N/A' else 0
                        change_percent_str = data.get('CHANGE_PERCENT', '0%').replace('%', '')
                        change_percent = float(change_percent_str) if change_percent_str != 'N/A' else 0
                        volume_str = data.get('VOLUME', '0').replace(',', '')
                        volume = int(volume_str) if volume_str != 'N/A' else 0
                    except (ValueError, AttributeError):
                        price = 0
                        change = 0
                        change_percent = 0
                        volume = 0
                    
                    return {
                        'symbol': data.get('SYMBOL', symbol),
                        'price': price,
                        'change': change,
                        'change_percent': change_percent,
                        'volume': volume,
                        'company_name': data.get('SYMBOL', symbol),
                        'market_cap': 'N/A',
                        'pe_ratio': 'N/A',
                        'timestamp': data.get('TIMESTAMP', format_ist_timestamp('%Y-%m-%d %H:%M:%S IST')),
                        'source': f"OpenAI Web Search ({data.get('MARKET', 'Unknown')})"
                    }
        except Exception as e:
            logger.warning(f"OpenAI web search failed for {symbol}: {str(e)}")
        return None

    async def get_stock_price(self, symbol: str, user_id: int = None) -> Optional[Dict]:
        """
        Get current stock price and basic info from multiple sources for ANY US stock
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL', 'TSLA', 'INFOSYS')
            user_id (int): Telegram user ID for OpenAI web search
            
        Returns:
            Dict with price data or None if error
        """
        try:
            original_symbol = symbol.upper().strip()
            symbol = self._normalize_symbol(original_symbol)
            logger.info(f"[DEBUG] Fetching price data for {symbol} (original: {original_symbol})")
            
            # Try multiple sources in order of preference (optimized for US stocks)
            sources = [
                ("Yahoo Finance US", lambda: self._try_yahoo_finance_global(original_symbol)),
                ("Yahoo Finance", lambda: self._try_yfinance(symbol)),
                ("Google Finance", lambda: self._try_google_finance(original_symbol)),
                ("Alpaca API", lambda: self._try_alpaca_api(symbol)),
                ("Alpha Vantage", lambda: self._try_alpha_vantage(symbol)),
            ]
            
            logger.info(f"[DEBUG] Attempting to fetch {symbol} from sources: {', '.join([s[0] for s in sources])}")
            
            # Add total timeout of 15 seconds for all traditional sources
            start_time = datetime.utcnow()  # Keep for elapsed time calculation
            
            for source_name, source_func in sources:
                try:
                    # Check if we've exceeded 15 seconds total
                    elapsed_time = (datetime.utcnow() - start_time).total_seconds()
                    if elapsed_time > 15:
                        logger.info(f"[DEBUG] 15-second timeout reached, moving to AI web search")
                        break
                        
                    logger.info(f"[DEBUG] Trying {source_name} for {symbol}")
                    result = await source_func()
                    if result:
                        logger.info(f"[DEBUG] Successfully got data from {source_name} for {symbol}")
                        logger.info(f"[DEBUG] Data for {symbol}: Price=${result.get('price', 'N/A')}, Change={result.get('change_percent', 'N/A')}%, Volume={result.get('volume', 'N/A')}")
                        logger.info(f"[DEBUG] Additional data: Company={result.get('company_name', 'N/A')}, High=${result.get('high', 'N/A')}, Low=${result.get('low', 'N/A')}")
                        return result
                except Exception as e:
                    logger.warning(f"[DEBUG] {source_name} failed for {symbol}: {str(e)}")
            
            # Try alternative symbol variations before AI web search
            alternative_symbols = [
                symbol + '.US',  # Some sources use .US suffix
                symbol + '.O',   # OTC markets
                symbol + '.Q',   # OTC markets
                symbol + '.PINK', # Pink sheets
                symbol + '.OTC'  # OTC markets
            ]
            
            logger.info(f"[DEBUG] Trying alternative symbols for {symbol}: {alternative_symbols}")
            
            for alt_symbol in alternative_symbols:
                try:
                    elapsed_time = (datetime.utcnow() - start_time).total_seconds()
                    if elapsed_time > 15:
                        logger.info(f"[DEBUG] Timeout reached during alternative symbol search")
                        break
                        
                    logger.info(f"[DEBUG] Trying alternative symbol: {alt_symbol}")
                    for source_name, source_func in sources:
                        try:
                            result = await source_func()
                            if result:
                                logger.info(f"[DEBUG] Successfully got data from {source_name} for {alt_symbol}")
                                logger.info(f"[DEBUG] Data for {alt_symbol}: Price=${result.get('price', 'N/A')}, Change={result.get('change_percent', 'N/A')}%, Volume={result.get('volume', 'N/A')}")
                                return result
                        except Exception as e:
                            continue
                except Exception as e:
                    continue
            
            logger.warning(f"[DEBUG] All primary sources failed for {symbol}")
            
            # Final fallback: OpenAI web search (this can find ANY stock)
            if user_id:
                logger.info(f"[DEBUG] All traditional sources failed! Triggering AI web search for {original_symbol}")
                result = await self._try_openai_web_search(original_symbol, user_id)
                if result:
                    logger.info(f"[DEBUG] AI web search SUCCESS for {original_symbol}!")
                    logger.info(f"[DEBUG] AI Data for {original_symbol}: Price=${result.get('price', 'N/A')}, Source={result.get('source', 'N/A')}")
                    return result
                else:
                    logger.warning(f"[DEBUG] AI web search also failed for {original_symbol}")
            
            logger.error(f"[DEBUG] No data found for {original_symbol} from ANY source (including AI web search)")
            return None
                
        except Exception as e:
            logger.error(f"[DEBUG] Error fetching price for {original_symbol}: {str(e)}")
            import traceback
            logger.error(f"[DEBUG] Full traceback: {traceback.format_exc()}")
            return None
    
    async def get_market_movers(self, limit: int = 10) -> Dict:
        """
        Get top gainers and losers in the market
        
        Args:
            limit (int): Number of stocks to return
            
        Returns:
            Dict with gainers and losers
        """
        try:
            logger.info("Fetching market movers")
            
            # Popular stocks to check
            popular_stocks = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
                'AMD', 'INTC', 'CRM', 'ORCL', 'ADBE', 'PYPL', 'DIS', 'BA',
                'JPM', 'BAC', 'WFC', 'GS', 'V', 'MA', 'JNJ', 'PFE', 'KO', 'PEP'
            ]
            
            movers_data = []
            
            # Fetch data for multiple stocks
            for symbol in popular_stocks[:20]:  # Limit to avoid rate limits
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period='1d')
                    
                    if not hist.empty and len(hist) > 1:
                        current = hist['Close'].iloc[-1]
                        # Use previous day's close, not today's open
                        previous = hist['Close'].iloc[-2] if len(hist) > 1 else hist['Open'].iloc[0]
                        change = current - previous
                        change_percent = (change / previous) * 100
                        
                        movers_data.append({
                            'symbol': symbol,
                            'price': round(float(current), 2),
                            'change': round(float(change), 2),
                            'change_percent': round(float(change_percent), 3),
                            'volume': int(hist['Volume'].iloc[-1]) if not pd.isna(hist['Volume'].iloc[-1]) else 0
                        })
                except Exception as e:
                    logger.warning(f"Failed to fetch data for {symbol}: {e}")
                    continue
            
            # Sort by change percentage
            movers_data.sort(key=lambda x: x['change_percent'], reverse=True)
            
            gainers = [stock for stock in movers_data if stock['change_percent'] > 0][:limit]
            losers = [stock for stock in movers_data if stock['change_percent'] < 0][-limit:]
            losers.reverse()  # Show worst performers first
            
            return {
                'gainers': gainers,
                'losers': losers,
                'timestamp': format_ist_timestamp('%Y-%m-%d %H:%M:%S IST')
            }
            
        except Exception as e:
            logger.error(f"Error fetching market movers: {e}")
            return {'gainers': [], 'losers': [], 'error': str(e)}
    
    async def get_sector_summary(self) -> Dict:
        """
        Get sector performance summary
        
        Returns:
            Dict with sector performance data
        """
        try:
            logger.info("Fetching sector summary")
            
            # Major sector ETFs
            sector_etfs = {
                'Technology': 'XLK',
                'Healthcare': 'XLV',
                'Financials': 'XLF',
                'Consumer Discretionary': 'XLY',
                'Communication Services': 'XLC',
                'Industrials': 'XLI',
                'Consumer Staples': 'XLP',
                'Energy': 'XLE',
                'Utilities': 'XLU',
                'Real Estate': 'XLRE',
                'Materials': 'XLB'
            }
            
            sector_data = []
            
            for sector_name, etf_symbol in sector_etfs.items():
                try:
                    ticker = yf.Ticker(etf_symbol)
                    hist = ticker.history(period='1d')
                    
                    if not hist.empty:
                        current = hist['Close'].iloc[-1]
                        # Use previous day's close, not today's open
                        previous = hist['Close'].iloc[-2] if len(hist) > 1 else hist['Open'].iloc[0]
                        change_percent = ((current - previous) / previous) * 100
                        
                        sector_data.append({
                            'sector': sector_name,
                            'symbol': etf_symbol,
                            'price': round(float(current), 2),
                            'change_percent': round(float(change_percent), 3),
                            'volume': int(hist['Volume'].iloc[-1]) if not pd.isna(hist['Volume'].iloc[-1]) else 0
                        })
                except Exception as e:
                    logger.warning(f"Failed to fetch data for {sector_name}: {e}")
                    continue
            
            # Sort by performance
            sector_data.sort(key=lambda x: x['change_percent'], reverse=True)
            
            return {
                'sectors': sector_data,
                'timestamp': format_ist_timestamp('%Y-%m-%d %H:%M:%S IST')
            }
            
        except Exception as e:
            logger.error(f"Error fetching sector summary: {e}")
            return {'sectors': [], 'error': str(e)}
    
    async def get_portfolio_summary(self) -> Dict:
        """
        Get portfolio summary from Alpaca (if available)
        
        Returns:
            Dict with portfolio data
        """
        try:
            if not self.alpaca_trading_client:
                return {
                    'error': 'Portfolio data requires trading API keys',
                    'message': 'Please configure your trading API credentials to view portfolio data.'
                }
            
            logger.info("Fetching portfolio summary")
            
            # Get account info
            account = self.alpaca_trading_client.get_account()
            
            # Get positions
            positions = self.alpaca_trading_client.list_positions()
            
            position_data = []
            total_market_value = 0
            
            for position in positions:
                market_value = float(position.market_value)
                total_market_value += market_value
                
                position_data.append({
                    'symbol': position.symbol,
                    'qty': float(position.qty),
                    'market_value': market_value,
                    'unrealized_pl': float(position.unrealized_pl),
                    'unrealized_plpc': float(position.unrealized_plpc) * 100,
                    'current_price': float(position.current_price)
                })
            
            return {
                'account_equity': float(account.equity),
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'total_market_value': total_market_value,
                'positions': position_data,
                'timestamp': format_ist_timestamp('%Y-%m-%d %H:%M:%S IST')
            }
            
        except Exception as e:
            logger.error(f"Error fetching portfolio summary: {e}")
            return {'error': str(e)}
    
    async def search_stocks(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Search for stocks by name or symbol
        
        Args:
            query (str): Search query
            limit (int): Number of results to return
            
        Returns:
            List of matching stocks
        """
        try:
            query = query.upper().strip()
            logger.info(f"Searching stocks for: {query}")
            
            # Common stock symbols for quick matching
            common_stocks = {
                'APPLE': 'AAPL', 'MICROSOFT': 'MSFT', 'GOOGLE': 'GOOGL', 'ALPHABET': 'GOOGL',
                'AMAZON': 'AMZN', 'TESLA': 'TSLA', 'META': 'META', 'FACEBOOK': 'META',
                'NVIDIA': 'NVDA', 'NETFLIX': 'NFLX', 'AMD': 'AMD', 'INTC': 'INTC',
                'SALESFORCE': 'CRM', 'ORACLE': 'ORCL', 'ADOBE': 'ADBE', 'PAYPAL': 'PYPL',
                'DISNEY': 'DIS', 'BOEING': 'BA', 'JPMORGAN': 'JPM', 'VISA': 'V',
                'MASTERCARD': 'MA', 'JOHNSON': 'JNJ', 'PFIZER': 'PFE', 'COCACOLA': 'KO',
                'PEPSI': 'PEP', 'WALMART': 'WMT', 'BERKSHIRE': 'BRK-B'
            }
            
            results = []
            
            # Direct symbol match
            if len(query) <= 5:
                try:
                    ticker = yf.Ticker(query)
                    info = ticker.info
                    if info and 'longName' in info:
                        results.append({
                            'symbol': query,
                            'name': info.get('longName', query),
                            'sector': info.get('sector', 'N/A'),
                            'market_cap': info.get('marketCap', 'N/A')
                        })
                except:
                    pass
            
            # Search common stocks
            for name, symbol in common_stocks.items():
                if query in name and len(results) < limit:
                    try:
                        ticker = yf.Ticker(symbol)
                        info = ticker.info
                        results.append({
                            'symbol': symbol,
                            'name': info.get('longName', name),
                            'sector': info.get('sector', 'N/A'),
                            'market_cap': info.get('marketCap', 'N/A')
                        })
                    except:
                        results.append({
                            'symbol': symbol,
                            'name': name,
                            'sector': 'N/A',
                            'market_cap': 'N/A'
                        })
            
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error searching stocks: {e}")
            return []
    
    async def get_market_news(self, limit: int = 5) -> List[Dict]:
        """
        Get latest market news (placeholder - would integrate with news API)
        
        Args:
            limit (int): Number of news items to return
            
        Returns:
            List of news items
        """
        try:
            # This would integrate with a news API in a real implementation
            # For now, return market status and general info
            return [
                {
                    'title': 'Market Update',
                    'summary': 'Real-time market data is available. Use commands like /price AAPL to get current stock prices.',
                    'timestamp': format_ist_timestamp('%Y-%m-%d %H:%M:%S IST'),
                    'source': 'Trading Bot'
                }
            ]
        except Exception as e:
            logger.error(f"Error fetching market news: {e}")
            return []

    def get_historical_data(self, symbol: str, period: str = "1y"):
        """Fetch historical OHLCV data for a symbol using Yahoo Finance."""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            if df.empty:
                return pd.DataFrame()
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            return df
        except Exception as e:
            print(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()