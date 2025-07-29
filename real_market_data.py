"""
Real Market Data Integration Service
Integrates multiple data sources for comprehensive market data
"""
import asyncio
import aiohttp
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False
    print("Warning: pandas not available in real_market_data")
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False
    print("Warning: numpy not available in real_market_data")
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, TYPE_CHECKING
if TYPE_CHECKING and PANDAS_AVAILABLE:
    import pandas as pd
import os
from logger import logger
from openai_service import OpenAIService

class RealMarketDataService:
    """Service for real-time market data from multiple sources"""
    
    def __init__(self):
        self.session = None
        self.cache = {}
        self.cache_duration = 60  # Cache for 60 seconds
        self.data_sources = {
            'yahoo': YahooFinanceSource(),
            'google_finance': GoogleFinanceSource(),
            'finnhub': FinnhubSource()
        }
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_stock_price(self, symbol: str, user_id: int = None) -> Dict:
        """Get real-time stock price from multiple sources"""
        try:
            # Try Yahoo Finance first (most reliable)
            price_data = await self.data_sources['yahoo'].get_price(symbol, self.session)
            if price_data and price_data.get('price') is not None:
                return price_data
                
            # Fallback to Google Finance
            price_data = await self.data_sources['google_finance'].get_price(symbol, self.session)
            if price_data and price_data.get('price') is not None:
                return price_data
                
            # Final fallback to Finnhub
            price_data = await self.data_sources['finnhub'].get_price(symbol, self.session)
            if price_data and price_data.get('price') is not None:
                return price_data
            
            logger.error(f"[DEBUG] No data found for {symbol} from ANY source")
            return None
            
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return None
    
    async def get_historical_data(self, symbol: str, period: str = "1y") -> Optional[Any]:
        """Get historical price data"""
        try:
            # Use Yahoo Finance for historical data
            df = await self.data_sources['yahoo'].get_historical(symbol, period, self.session)
            return df
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return None if not PANDAS_AVAILABLE else pd.DataFrame()
    
    async def get_market_data(self, symbol: str) -> Dict:
        """Get comprehensive market data including price, volume, indicators"""
        try:
            # Get real-time price
            price_data = await self.get_stock_price(symbol)
            if not price_data:
                return None
            
            # Get historical data for indicators
            hist_data = await self.get_historical_data(symbol, "3mo")
            
            if not PANDAS_AVAILABLE or hist_data is None or (hasattr(hist_data, 'empty') and hist_data.empty):
                return price_data
            
            # Calculate technical indicators
            indicators = self.calculate_technical_indicators(hist_data)
            
            # Combine data
            market_data = {
                **price_data,
                'indicators': indicators,
                'historical_data': hist_data.tail(30).to_dict('records') if PANDAS_AVAILABLE and hist_data is not None else []
            }
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return None
    
    def calculate_technical_indicators(self, df: Optional[Any]) -> Dict:
        """Calculate comprehensive technical indicators"""
        try:
            if not PANDAS_AVAILABLE or df is None or (hasattr(df, 'empty') and df.empty):
                return {}
            
            indicators = {}
            
            # Moving Averages
            indicators['sma_20'] = df['Close'].rolling(window=20).mean().iloc[-1]
            indicators['sma_50'] = df['Close'].rolling(window=50).mean().iloc[-1]
            indicators['ema_12'] = df['Close'].ewm(span=12).mean().iloc[-1]
            indicators['ema_26'] = df['Close'].ewm(span=26).mean().iloc[-1]
            
            # MACD
            ema_12 = df['Close'].ewm(span=12).mean()
            ema_26 = df['Close'].ewm(span=26).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9).mean()
            indicators['macd'] = macd_line.iloc[-1]
            indicators['macd_signal'] = signal_line.iloc[-1]
            indicators['macd_histogram'] = macd_line.iloc[-1] - signal_line.iloc[-1]
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['rsi'] = 100 - (100 / (1 + rs.iloc[-1]))
            
            # Bollinger Bands
            sma_20 = df['Close'].rolling(window=20).mean()
            std_20 = df['Close'].rolling(window=20).std()
            indicators['bb_upper'] = sma_20.iloc[-1] + (std_20.iloc[-1] * 2)
            indicators['bb_middle'] = sma_20.iloc[-1]
            indicators['bb_lower'] = sma_20.iloc[-1] - (std_20.iloc[-1] * 2)
            
            # Volume indicators
            indicators['volume_sma'] = df['Volume'].rolling(window=20).mean().iloc[-1]
            indicators['volume_ratio'] = df['Volume'].iloc[-1] / indicators['volume_sma']
            
            # Price momentum
            indicators['price_change_1d'] = ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
            indicators['price_change_5d'] = ((df['Close'].iloc[-1] - df['Close'].iloc[-6]) / df['Close'].iloc[-6]) * 100
            indicators['price_change_20d'] = ((df['Close'].iloc[-1] - df['Close'].iloc[-21]) / df['Close'].iloc[-21]) * 100
            
            # Support and Resistance
            indicators['support'] = df['Low'].rolling(window=20).min().iloc[-1]
            indicators['resistance'] = df['High'].rolling(window=20).max().iloc[-1]
            
            # ATR (Average True Range)
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            indicators['atr'] = true_range.rolling(window=14).mean().iloc[-1]
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return {}
    

    
    async def get_earnings_calendar(self, symbol: str) -> List[Dict]:
        """Get upcoming earnings calendar"""
        try:
            # Use Yahoo Finance for earnings data
            earnings = await self.data_sources['yahoo'].get_earnings(symbol, self.session)
            return earnings
        except Exception as e:
            logger.error(f"Error getting earnings for {symbol}: {e}")
            return []

class YahooFinanceSource:
    """Yahoo Finance data source"""
    
    async def get_price(self, symbol: str, session: aiohttp.ClientSession) -> Dict:
        """Get real-time price from Yahoo Finance with retry logic"""
        import asyncio
        
        for attempt in range(3):  # Try up to 3 times
            try:
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
                params = {
                    'interval': '1m',
                    'range': '1d'
                }
                
                # Add delay between attempts to avoid rate limiting
                if attempt > 0:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
                            result = data['chart']['result'][0]
                            meta = result.get('meta', {})
                            
                            # Only return data if we have a valid price
                            price = meta.get('regularMarketPrice')
                            if price is not None and price > 0:
                                return {
                                    'price': price,
                                    'change': meta.get('regularMarketChange', 0),
                                    'change_percent': meta.get('regularMarketChangePercent', 0),
                                    'volume': meta.get('regularMarketVolume', 0),
                                    'high': meta.get('regularMarketDayHigh', 0),
                                    'low': meta.get('regularMarketDayLow', 0),
                                    'open': meta.get('regularMarketOpen', 0),
                                    'previous_close': meta.get('previousClose', 0),
                                    'source': 'yahoo'
                                }
                    elif response.status == 429:  # Rate limited
                        logger.warning(f"Yahoo Finance rate limited for {symbol}, attempt {attempt + 1}")
                        if attempt < 2:  # Don't sleep on last attempt
                            continue
                    else:
                        logger.warning(f"Yahoo Finance returned status {response.status} for {symbol}")
                        break  # Don't retry for other HTTP errors
                
            except Exception as e:
                logger.error(f"Yahoo Finance error for {symbol} (attempt {attempt + 1}): {e}")
                if attempt < 2:  # Don't sleep on last attempt
                    await asyncio.sleep(1)
        
        return None
    
    async def get_historical(self, symbol: str, period: str, session: aiohttp.ClientSession) -> Optional[Any]:
        """Get historical data from Yahoo Finance with rate limiting handling"""
        import asyncio
        
        # Try multiple times with exponential backoff for rate limiting
        for attempt in range(3):
            try:
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
                params = {
                    'interval': '1d',
                    'range': period
                }
                
                # Add proper headers to avoid rate limiting
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'application/json',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1'
                }
                
                # Add delay between attempts to avoid rate limiting
                if attempt > 0:
                    delay = 2 ** attempt  # Exponential backoff: 2, 4 seconds
                    await asyncio.sleep(delay)
                
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
                            result = data['chart']['result'][0]
                            
                            if 'timestamp' in result and 'indicators' in result:
                                timestamps = result['timestamp']
                                quotes = result['indicators']['quote'][0]
                                
                                if not PANDAS_AVAILABLE:
                                    return None
                                df = pd.DataFrame({
                                    'Date': pd.to_datetime(timestamps, unit='s', utc=True),
                                    'Open': quotes['open'],
                                    'High': quotes['high'],
                                    'Low': quotes['low'],
                                    'Close': quotes['close'],
                                    'Volume': quotes['volume']
                                })
                                
                                return df.dropna()
                            
                    elif response.status == 429:  # Rate limited
                        logger.warning(f"Yahoo Finance rate limited for {symbol}, attempt {attempt + 1}/3")
                        if attempt < 2:  # Don't continue on last attempt
                            continue
                    else:
                        logger.warning(f"Yahoo Finance returned status {response.status} for {symbol}")
                        break  # Don't retry for other HTTP errors
                
            except Exception as e:
                logger.error(f"Yahoo Finance error for {symbol} (attempt {attempt + 1}): {e}")
                if attempt < 2:  # Don't sleep on last attempt
                    await asyncio.sleep(1)
        return None if not PANDAS_AVAILABLE else pd.DataFrame()
    
    async def get_earnings(self, symbol: str, session: aiohttp.ClientSession) -> List[Dict]:
        """Get earnings calendar from Yahoo Finance"""
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            params = {
                'interval': '1d',
                'range': '1y',
                'includePrePost': 'false'
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
                        result = data['chart']['result'][0]
                        events = result.get('events', {})
                        earnings = events.get('earnings', {})
                        
                        earnings_list = []
                        for timestamp, earning in earnings.items():
                            earnings_list.append({
                                'date': datetime.fromtimestamp(int(timestamp)) if not PANDAS_AVAILABLE else pd.to_datetime(int(timestamp), unit='s', utc=True),
                                'estimate': earning.get('estimate', {}).get('raw', 0),
                                'actual': earning.get('actual', {}).get('raw', 0)
                            })
                        
                        return earnings_list
            
            return []
            
        except Exception as e:
            logger.error(f"Yahoo Finance earnings error for {symbol}: {e}")
            return []

class GoogleFinanceSource:
    """Alternative data source using Yahoo Finance v7 API"""
    
    async def get_price(self, symbol: str, session: aiohttp.ClientSession) -> Dict:
        """Get real-time price from Yahoo Finance v7 API"""
        import asyncio
        
        for attempt in range(2):  # Try up to 2 times
            try:
                # Use Yahoo Finance v7 quote API (different endpoint)
                url = f"https://query1.finance.yahoo.com/v7/finance/quote"
                params = {
                    'symbols': symbol
                }
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                }
                
                if attempt > 0:
                    await asyncio.sleep(2)
                
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'quoteResponse' in data and 'result' in data['quoteResponse'] and data['quoteResponse']['result']:
                            quote = data['quoteResponse']['result'][0]
                            
                            # Only return data if we have a valid price
                            price = quote.get('regularMarketPrice')
                            if price is not None and price > 0:
                                return {
                                    'price': price,
                                    'change': quote.get('regularMarketChange', 0),
                                    'change_percent': quote.get('regularMarketChangePercent', 0),
                                    'volume': quote.get('regularMarketVolume', 0),
                                    'high': quote.get('regularMarketDayHigh', 0),
                                    'low': quote.get('regularMarketDayLow', 0),
                                    'open': quote.get('regularMarketOpen', 0),
                                    'previous_close': quote.get('regularMarketPreviousClose', 0),
                                    'source': 'yahoo_v7'
                                }
                    elif response.status == 429:
                        logger.warning(f"Yahoo v7 API rate limited for {symbol}, attempt {attempt + 1}")
                        if attempt < 1:
                            continue
                    else:
                        logger.warning(f"Yahoo v7 API returned status {response.status} for {symbol}")
                        break
                
            except Exception as e:
                logger.error(f"Yahoo v7 API error for {symbol} (attempt {attempt + 1}): {e}")
                if attempt < 1:
                    await asyncio.sleep(1)
        
        return None
    


class FinnhubSource:
    """Fallback data source using Yahoo Finance search API"""
    
    async def get_price(self, symbol: str, session: aiohttp.ClientSession) -> Dict:
        """Get real-time price from Yahoo Finance search API as fallback"""
        import asyncio
        
        for attempt in range(2):  # Try up to 2 times
            try:
                # Use Yahoo Finance search API as fallback
                url = f"https://query1.finance.yahoo.com/v1/finance/search"
                params = {
                    'q': symbol,
                    'quotesCount': 1,
                    'newsCount': 0
                }
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                }
                
                if attempt > 0:
                    await asyncio.sleep(3)
                
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'quotes' in data and data['quotes']:
                            quote = data['quotes'][0]
                            
                            # Only return data if we have a valid current price
                            current_price = quote.get('regularMarketPrice')
                            if current_price is not None and current_price > 0:
                                previous_close = quote.get('regularMarketPreviousClose', current_price)
                                change = current_price - previous_close
                                change_percent = (change / previous_close * 100) if previous_close > 0 else 0
                                
                                return {
                                    'price': current_price,
                                    'change': change,
                                    'change_percent': change_percent,
                                    'volume': quote.get('regularMarketVolume', 0),
                                    'high': quote.get('regularMarketDayHigh', current_price),
                                    'low': quote.get('regularMarketDayLow', current_price),
                                    'open': quote.get('regularMarketOpen', current_price),
                                    'previous_close': previous_close,
                                    'source': 'yahoo_search'
                                }
                    elif response.status == 429:
                        logger.warning(f"Yahoo search API rate limited for {symbol}, attempt {attempt + 1}")
                        if attempt < 1:
                            continue
                    else:
                        logger.warning(f"Yahoo search API returned status {response.status} for {symbol}")
                        break
                
            except Exception as e:
                logger.error(f"Yahoo search API error for {symbol} (attempt {attempt + 1}): {e}")
                if attempt < 1:
                    await asyncio.sleep(1)
        
        return None