"""
Trading Intelligence Service - Advanced AI-powered trading analysis
Combines market data with AI insights for intelligent trading decisions
"""
import os
import asyncio
import json
from datetime import datetime, timedelta
from timezone_utils import format_ist_timestamp
from typing import Dict, List, Optional, Tuple
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
from logger import logger
from openai_service import OpenAIService
from market_data_service import MarketDataService

class TradingIntelligence:
    """Advanced AI-powered trading analysis and recommendations"""
    
    def __init__(self):
        """Initialize trading intelligence service"""
        self.openai_service = OpenAIService()
        self.market_service = MarketDataService()
        logger.info("Trading intelligence service initialized")
    
    async def analyze_stock(self, symbol: str, user_id: int = None) -> Dict:
        """
        Perform comprehensive AI analysis of a stock with enhanced technical insights
        
        Args:
            symbol (str): Stock symbol to analyze
            user_id (int): Telegram user ID for AI web search fallback
            
        Returns:
            Dict with comprehensive analysis including price trends, technical indicators, and recommendations
        """
        try:
            symbol = symbol.upper().strip()
            logger.info(f"Performing enhanced comprehensive analysis for {symbol}")
            
            # Get market data
            price_data = await self.market_service.get_stock_price(symbol, user_id)
            if not price_data:
                return {'error': f'Could not fetch data for {symbol}'}
            
            # Extract key price metrics for analysis
            current_price = price_data.get('price', 0)
            change_percent = price_data.get('change_percent', 0)
            volume = price_data.get('volume', 0)
            high_52w = price_data.get('high_52w', current_price)
            low_52w = price_data.get('low_52w', current_price)
            market_cap = price_data.get('market_cap', 'N/A')
            pe_ratio = price_data.get('pe_ratio', 'N/A')
            company_name = price_data.get('company_name', symbol)
            
            # Calculate technical levels (simplified)
            price_range = high_52w - low_52w if high_52w and low_52w else 0
            support_level = current_price - (price_range * 0.05) if price_range > 0 else current_price * 0.95
            resistance_level = current_price + (price_range * 0.05) if price_range > 0 else current_price * 1.05
            
            # Determine trend direction based on actual data
            if change_percent > 2:
                trend = "Strong Bullish"
                trend_emoji = "🚀"
            elif change_percent > 0.5:
                trend = "Bullish"
                trend_emoji = "📈"
            elif change_percent < -2:
                trend = "Strong Bearish"
                trend_emoji = "📉"
            elif change_percent < -0.5:
                trend = "Bearish"
                trend_emoji = "🔻"
            else:
                trend = "Neutral"
                trend_emoji = "➡️"
            
            # Calculate position in 52-week range
            if high_52w and low_52w and high_52w != low_52w:
                range_position = ((current_price - low_52w) / (high_52w - low_52w)) * 100
            else:
                range_position = 50  # Default to middle
            
            # Determine volume analysis
            volume_analysis = "High" if volume > 1000000 else "Average" if volume > 100000 else "Low"
            
            # Calculate dynamic recommendation thresholds
            price_momentum = abs(change_percent)
            volatility_factor = (high_52w - low_52w) / current_price if current_price > 0 else 0.1
            
            # Dynamic stop-loss calculation based on volatility
            if volatility_factor > 0.5:  # High volatility stock
                stop_loss_pct = 0.08  # 8% stop loss
                risk_level = "High"
            elif volatility_factor > 0.3:  # Medium volatility
                stop_loss_pct = 0.06  # 6% stop loss
                risk_level = "Medium"
            else:  # Low volatility
                stop_loss_pct = 0.04  # 4% stop loss
                risk_level = "Low"
            
            stop_loss_level = current_price * (1 - stop_loss_pct)
            invalidation_level = support_level * 0.97  # 3% below support
            
            # Generate enhanced AI analysis with comprehensive prompt including actual data
            analysis_prompt = f"""
            Provide a comprehensive trading analysis for {symbol} ({company_name}) using this REAL market data:
            
            📊 CURRENT MARKET DATA:
            • Price: ${current_price:.2f} ({change_percent:+.2f}%)
            • Trend: {trend} {trend_emoji}
            • Volume: {volume:,} shares ({volume_analysis})
            • 52W Range: ${low_52w:.2f} - ${high_52w:.2f}
            • Position in Range: {range_position:.1f}%
            • Market Cap: {market_cap}
            • P/E Ratio: {pe_ratio}
            • Volatility: {risk_level} (52W range: {volatility_factor:.1%})
            
            📈 TECHNICAL LEVELS:
            • Support: ~${support_level:.2f}
            • Resistance: ~${resistance_level:.2f}
            • Suggested Stop-Loss: ${stop_loss_level:.2f} ({stop_loss_pct:.0%} risk)
            • Trend Invalidation: ${invalidation_level:.2f}
            
            Based on this SPECIFIC data for {symbol}, provide:
            
            TECHNICAL ANALYSIS:
            - Current price action and momentum specific to {symbol}
            - Why these support/resistance levels matter for this stock
            - Volume significance in {symbol}'s context
            - What the {range_position:.1f}% position in 52W range means
            
            RECOMMENDATION:
            Provide a clear BUY/SELL/HOLD decision with:
            - Specific reasoning based on {symbol}'s current data
            - Conditional scenarios: "If price breaks above ${resistance_level:.2f}, consider..."
            - "If price falls below ${support_level:.2f}, then..."
            - Entry/exit strategy tailored to this stock's behavior
            
            RISK MANAGEMENT:
            - Why stop-loss is set at ${stop_loss_level:.2f} (explain the {stop_loss_pct:.0%} logic)
            - Trend invalidation at ${invalidation_level:.2f} - what this means
            - Position sizing recommendations for {risk_level} volatility
            - When to reassess the position
            
            OUTLOOK & CATALYSTS:
            - Short-term expectations specific to {symbol}
            - Key price levels that could trigger moves
            - Sector/company-specific factors to watch
            
            IMPORTANT: 
            - Avoid generic phrases like "mixed signals" - be specific to {symbol}
            - Use varied language and unique insights for each stock
            - Base ALL analysis on the actual data provided
            - Keep under 1500 characters but make it actionable and personalized
            """
            
            # Add debug logging for the prompt and data
            logger.info(f"Analysis prompt for {symbol}: Price=${current_price}, Change={change_percent}%, Volume={volume}")
            
            ai_analysis = await self.openai_service.generate_response(analysis_prompt, 0)
            
            # Check if analysis failed or returned an error message
            if not ai_analysis or ai_analysis == "ANALYSIS_FAILED" or len(ai_analysis.strip()) < 10:
                logger.error(f"OpenAI returned invalid analysis for {symbol}: {ai_analysis}")
                # Provide enhanced fallback analysis
                ai_analysis = self._generate_fallback_analysis(symbol, price_data, trend, support_level, resistance_level, trend_emoji)
                
            # Provide a fallback analysis if the AI response contains error indicators
            error_phrases = ['I encountered an issue', 'Please try again later', 'I apologize', 'I\'m sorry', 
                           'error', 'unable to', 'cannot', 'couldn\'t', 'failed', 'timeout']
            
            if any(phrase in ai_analysis.lower() for phrase in error_phrases):
                logger.warning(f"AI analysis for {symbol} contains error phrases, providing enhanced fallback analysis")
                ai_analysis = self._generate_fallback_analysis(symbol, price_data, trend, support_level, resistance_level, trend_emoji)
            
            return {
                'symbol': symbol,
                'current_data': price_data,
                'technical_levels': {
                    'support': support_level,
                    'resistance': resistance_level,
                    'trend': trend,
                    'trend_emoji': trend_emoji,
                    'range_position': range_position
                },
                'ai_analysis': ai_analysis,
                'timestamp': format_ist_timestamp('%Y-%m-%d %H:%M:%S IST')
            }
            
        except Exception as e:
            logger.error(f"Error analyzing stock {symbol}: {e}")
            return {'error': str(e)}
    
    def _generate_fallback_analysis(self, symbol: str, price_data: Dict, trend: str, support: float, resistance: float, trend_emoji: str) -> str:
        """
        Generate a comprehensive fallback analysis when AI fails
        
        Args:
            symbol (str): Stock symbol
            price_data (Dict): Market data
            trend (str): Trend direction
            support (float): Support level
            resistance (float): Resistance level
            trend_emoji (str): Trend emoji
            
        Returns:
            str: Formatted fallback analysis
        """
        current_price = price_data.get('price', 0)
        change_percent = price_data.get('change_percent', 0)
        volume = price_data.get('volume', 0)
        company_name = price_data.get('company_name', symbol)
        high_52w = price_data.get('high_52w', current_price)
        low_52w = price_data.get('low_52w', current_price)
        
        # Calculate volatility and risk metrics
        volatility_factor = (high_52w - low_52w) / current_price if current_price > 0 else 0.1
        price_momentum = abs(change_percent)
        
        # Dynamic stop-loss based on volatility
        if volatility_factor > 0.5:
            stop_loss_pct = 0.08
            risk_level = "High"
            volatility_desc = "highly volatile"
        elif volatility_factor > 0.3:
            stop_loss_pct = 0.06
            risk_level = "Medium"
            volatility_desc = "moderately volatile"
        else:
            stop_loss_pct = 0.04
            risk_level = "Low"
            volatility_desc = "relatively stable"
        
        stop_loss_level = current_price * (1 - stop_loss_pct)
        invalidation_level = support * 0.97
        
        # Calculate 52-week position
        if high_52w and low_52w and high_52w != low_52w:
            range_position = ((current_price - low_52w) / (high_52w - low_52w)) * 100
        else:
            range_position = 50
        
        # Dynamic recommendation logic with varied language
        if "Strong Bullish" in trend and change_percent > 2:
            recommendation = "BUY"
            reasoning = f"{symbol} demonstrates powerful upward momentum with {change_percent:+.2f}% surge"
            conditional = f"If {symbol} sustains above ${resistance:.2f}, expect continuation to higher targets"
            strategy = "Consider scaling into position on any minor pullbacks"
        elif "Bullish" in trend and change_percent > 0.5:
            recommendation = "BUY"
            reasoning = f"Positive momentum building in {symbol} with solid {change_percent:+.2f}% advance"
            conditional = f"Breakout above ${resistance:.2f} would signal accelerated uptrend"
            strategy = "Enter on dips near support levels for better risk/reward"
        elif "Strong Bearish" in trend and change_percent < -2:
            recommendation = "SELL/AVOID"
            reasoning = f"{symbol} under significant selling pressure with {change_percent:.2f}% decline"
            conditional = f"Break below ${support:.2f} could trigger further downside to ${support * 0.95:.2f}"
            strategy = "Exit positions or avoid new entries until trend stabilizes"
        elif "Bearish" in trend and change_percent < -0.5:
            recommendation = "HOLD/CAUTION"
            reasoning = f"Weakness emerging in {symbol} with {change_percent:.2f}% drop"
            conditional = f"Watch ${support:.2f} support closely - break would confirm bearish bias"
            strategy = "Reduce position size or wait for clearer directional signals"
        else:
            recommendation = "HOLD"
            reasoning = f"{symbol} consolidating with {change_percent:+.2f}% move - awaiting catalyst"
            conditional = f"Range-bound between ${support:.2f} support and ${resistance:.2f} resistance"
            strategy = "Monitor for breakout direction before committing capital"
        
        # Varied volume analysis
        if volume > 2000000:
            volume_desc = "exceptional trading interest"
        elif volume > 1000000:
            volume_desc = "strong institutional activity"
        elif volume > 500000:
            volume_desc = "healthy participation"
        else:
            volume_desc = "light trading volume"
        
        # Position in range analysis
        if range_position > 80:
            range_desc = f"near 52-week highs ({range_position:.0f}% of range)"
        elif range_position > 60:
            range_desc = f"in upper range ({range_position:.0f}% of 52W range)"
        elif range_position > 40:
            range_desc = f"mid-range position ({range_position:.0f}% of 52W range)"
        elif range_position > 20:
            range_desc = f"lower range territory ({range_position:.0f}% of 52W range)"
        else:
            range_desc = f"near 52-week lows ({range_position:.0f}% of range)"
        
        return f"""TECHNICAL ANALYSIS for {symbol}:
{trend_emoji} {symbol} showing {trend.lower()} pattern with {volume_desc}
Trading at ${current_price:.2f}, positioned {range_desc}
Key levels: Support ${support:.2f} | Resistance ${resistance:.2f}

RECOMMENDATION: {recommendation}
{reasoning}

CONDITIONAL STRATEGY:
{conditional}
{strategy}

RISK MANAGEMENT:
Stop-loss: ${stop_loss_level:.2f} ({stop_loss_pct:.0%} risk for {volatility_desc} stock)
Trend invalidation: ${invalidation_level:.2f} (reassess if breached)
Volatility: {risk_level} - adjust position size accordingly

NEXT STEPS:
Monitor price action around ${support:.2f}-${resistance:.2f} range
Watch for volume confirmation on any directional moves"""
    

    
    async def generate_watchlist_recommendations(self, risk_level: str = 'medium', user_id: int = None) -> Dict:
        """
        Generate AI-powered watchlist recommendations
        
        Args:
            risk_level (str): 'low', 'medium', 'high'
            user_id (int): Telegram user ID for AI web search fallback
            
        Returns:
            Dict with watchlist recommendations
        """
        try:
            logger.info(f"Generating watchlist recommendations for {risk_level} risk")
            
            # Get current market data
            movers = await self.market_service.get_market_movers(limit=15)
            sector_data = await self.market_service.get_sector_summary()
            
            # Define stock pools by risk level
            risk_pools = {
                'low': ['AAPL', 'MSFT', 'GOOGL', 'JNJ', 'PG', 'KO', 'V', 'MA'],
                'medium': ['AMZN', 'TSLA', 'NVDA', 'META', 'CRM', 'ADBE', 'NFLX', 'DIS'],
                'high': ['AMD', 'PLTR', 'COIN', 'ROKU', 'SQ', 'SHOP', 'ZM', 'PTON']
            }
            
            stock_pool = risk_pools.get(risk_level, risk_pools['medium'])
            
            # Analyze each stock in the pool
            recommendations = []
            for symbol in stock_pool[:5]:  # Limit to 5 to avoid rate limits
                try:
                    price_data = await self.market_service.get_stock_price(symbol, user_id)
                    if price_data:
                        recommendations.append({
                            'symbol': symbol,
                            'current_price': price_data.get('price'),
                            'change_percent': price_data.get('change_percent'),
                            'volume': price_data.get('volume'),
                            'market_cap': price_data.get('market_cap')
                        })
                except Exception as e:
                    logger.warning(f"Failed to get data for {symbol}: {e}")
            
            # Generate AI recommendations
            rec_prompt = f"""
            Based on current market conditions and {risk_level} risk tolerance, analyze these stocks for a watchlist:
            
            {json.dumps(recommendations, indent=2)}
            
            Current market performance: {sector_data.get('sectors', [{}])[0].get('change_percent', 0):.1f}% average sector performance
            
            For each stock, provide:
            1. Why it's suitable for {risk_level} risk
            2. Entry point suggestion
            3. Key levels to watch
            4. Short-term catalyst potential
            
            Rank them 1-5 by attractiveness and keep total response under 1200 characters.
            """
            
            ai_recommendations = await self.openai_service.generate_response(rec_prompt, 0)
            
            return {
                'risk_level': risk_level,
                'recommendations': recommendations,
                'ai_analysis': ai_recommendations,
                'market_context': sector_data,
                'timestamp': format_ist_timestamp('%Y-%m-%d %H:%M:%S IST')
            }
            
        except Exception as e:
            logger.error(f"Error generating watchlist recommendations: {e}")
            return {'error': str(e)}
    
    async def analyze_portfolio_risk(self, portfolio_data: Dict) -> Dict:
        """
        Analyze portfolio risk and provide optimization suggestions
        
        Args:
            portfolio_data (Dict): Portfolio data from Alpaca
            
        Returns:
            Dict with risk analysis and suggestions
        """
        try:
            if not portfolio_data.get('positions'):
                return {'error': 'No portfolio positions to analyze'}
            
            logger.info("Analyzing portfolio risk")
            
            positions = portfolio_data['positions']
            total_value = portfolio_data.get('total_market_value', 0)
            
            # Calculate portfolio metrics
            position_weights = []
            sector_exposure = {}
            risk_metrics = []
            
            for position in positions:
                weight = (position['market_value'] / total_value) * 100 if total_value > 0 else 0
                position_weights.append({
                    'symbol': position['symbol'],
                    'weight': weight,
                    'unrealized_plpc': position['unrealized_plpc']
                })
            
            # Generate AI risk analysis
            risk_prompt = f"""
            Analyze this portfolio for risk assessment:
            
            Total Portfolio Value: ${total_value:,.2f}
            Number of Positions: {len(positions)}
            
            Position Weights:
            {json.dumps(position_weights, indent=2)}
            
            Current P&L: {sum([p['unrealized_pl'] for p in positions]):.2f}
            
            Provide:
            1. Diversification assessment
            2. Concentration risk (any position >20%)
            3. Sector/correlation risks
            4. Rebalancing suggestions
            5. Risk score (1-10, 10 being highest risk)
            
            Keep response under 1000 characters with actionable advice.
            """
            
            risk_analysis = await self.openai_service.generate_response(risk_prompt, 0)
            
            # Calculate simple risk metrics
            max_weight = max([p['weight'] for p in position_weights]) if position_weights else 0
            num_positions = len(positions)
            concentration_risk = "High" if max_weight > 25 else "Medium" if max_weight > 15 else "Low"
            
            return {
                'portfolio_value': total_value,
                'position_count': num_positions,
                'max_position_weight': max_weight,
                'concentration_risk': concentration_risk,
                'position_weights': position_weights,
                'ai_risk_analysis': risk_analysis,
                'timestamp': format_ist_timestamp('%Y-%m-%d %H:%M:%S IST')
            }
            
        except Exception as e:
            logger.error(f"Error analyzing portfolio risk: {e}")
            return {'error': str(e)}
    
    async def get_trading_opportunities(self) -> Dict:
        """
        Identify current trading opportunities using AI analysis
        
        Returns:
            Dict with trading opportunities
        """
        try:
            logger.info("Identifying trading opportunities")
            
            # Get market data
            movers = await self.market_service.get_market_movers(limit=20)
            sector_data = await self.market_service.get_sector_summary()
            
            # Identify patterns
            strong_gainers = [g for g in movers.get('gainers', []) if g['change_percent'] > 3]
            strong_losers = [l for l in movers.get('losers', []) if l['change_percent'] < -3]
            high_volume = [s for s in movers.get('gainers', []) + movers.get('losers', []) if s['volume'] > 1000000]
            
            # Generate AI opportunity analysis
            opp_prompt = f"""
            Identify trading opportunities based on current market data:
            
            Strong Gainers (>3%): {', '.join([f"{g['symbol']} (+{g['change_percent']:.1f}%)" for g in strong_gainers[:5]])}
            Strong Losers (<-3%): {', '.join([f"{l['symbol']} ({l['change_percent']:.1f}%)" for l in strong_losers[:5]])}
            High Volume Stocks: {', '.join([f"{h['symbol']}" for h in high_volume[:5]])}
            
            Best Performing Sector: {sector_data.get('sectors', [{}])[0].get('sector', 'N/A')} ({sector_data.get('sectors', [{}])[0].get('change_percent', 0):.1f}%)
            Worst Performing Sector: {sector_data.get('sectors', [{}])[-1].get('sector', 'N/A')} ({sector_data.get('sectors', [{}])[-1].get('change_percent', 0):.1f}%)
            
            Provide:
            1. Top 3 momentum plays (for day trading)
            2. Top 2 dip buying opportunities
            3. Sector rotation opportunities
            4. Risk warnings for current market
            
            Keep response under 1000 characters with specific entry strategies.
            """
            
            opportunities = await self.openai_service.generate_response(opp_prompt, 0)
            
            return {
                'strong_gainers': strong_gainers,
                'strong_losers': strong_losers,
                'high_volume': high_volume,
                'sector_leaders': sector_data.get('sectors', [])[:3],
                'sector_laggards': sector_data.get('sectors', [])[-3:],
                'ai_opportunities': opportunities,
                'timestamp': format_ist_timestamp('%Y-%m-%d %H:%M:%S IST')
            }
            
        except Exception as e:
            logger.error(f"Error identifying trading opportunities: {e}")
            return {'error': str(e)}