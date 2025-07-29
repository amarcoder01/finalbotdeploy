#!/usr/bin/env python3
"""
Debug script to identify why signal strength is showing 0.0%
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from telegram_handler import TelegramHandler
from market_data_service import MarketDataService
from openai_service import OpenAIService
from config import Config
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def debug_signal_strength():
    """Debug the signal strength calculation for a specific symbol"""
    try:
        # Create telegram handler instance (it initializes all services internally)
        telegram_handler = TelegramHandler()
        
        # Test symbol that's showing STRONG SELL with 0.0%
        symbol = "TSLA"  # You can change this to the problematic symbol
        
        print(f"\n=== DEBUGGING SIGNAL STRENGTH FOR {symbol} ===")
        
        # Get comprehensive analysis data
        analysis_data = await telegram_handler._get_comprehensive_deep_analysis(symbol, "3mo")
        
        if not analysis_data:
            print(f"❌ Failed to get analysis data for {symbol}")
            return
        
        indicators = analysis_data.get('indicators', {})
        sentiment = analysis_data.get('sentiment', {})
        risk_metrics = analysis_data.get('risk_metrics', {})
        
        print(f"\n📊 INDICATORS DATA:")
        print(f"RSI: {indicators.get('rsi', 'N/A')}")
        print(f"MACD Line: {indicators.get('macd_line', 'N/A')}")
        print(f"MACD Signal: {indicators.get('macd_signal', 'N/A')}")
        
        print(f"\n🎭 SENTIMENT DATA:")
        print(f"Overall Score: {sentiment.get('overall_score', 'N/A')}")
        print(f"Label: {sentiment.get('label', 'N/A')}")
        print(f"Confidence: {sentiment.get('confidence', 'N/A')}")
        print(f"News Sentiment: {sentiment.get('news_sentiment', 'N/A')}")
        print(f"Social Sentiment: {sentiment.get('social_sentiment', 'N/A')}")
        
        print(f"\n⚠️ RISK METRICS:")
        print(f"Sharpe Ratio: {risk_metrics.get('sharpe_ratio', 'N/A')}")
        print(f"Volatility Annual: {risk_metrics.get('volatility_annual', 'N/A')}")
        print(f"Beta: {risk_metrics.get('beta', 'N/A')}")
        
        # Calculate signal strength step by step
        print(f"\n🔍 SIGNAL STRENGTH CALCULATION:")
        
        score = 0.5  # Start neutral
        print(f"Starting score: {score}")
        
        # RSI contribution (20%) - FIXED to match actual logic
        rsi = indicators.get('rsi')
        if rsi is not None:
            if rsi < 30:
                score += 0.2
                print(f"RSI {rsi} < 30 (oversold): +0.2 → {score}")
            elif rsi > 70:
                score -= 0.2
                print(f"RSI {rsi} > 70 (overbought): -0.2 → {score}")
            else:
                print(f"RSI {rsi} (neutral): no change → {score}")
        else:
            print(f"RSI missing: no change → {score}")
        
        # MACD contribution (20%) - FIXED to match actual logic
        macd_line = indicators.get('macd_line')
        macd_signal = indicators.get('macd_signal')
        
        if macd_line is not None and macd_signal is not None:
            if macd_line > macd_signal:
                score += 0.2
                print(f"MACD {macd_line} > {macd_signal}: +0.2 → {score}")
            else:
                score -= 0.2
                print(f"MACD {macd_line} <= {macd_signal}: -0.2 → {score}")
        else:
            print(f"MACD missing (line={macd_line}, signal={macd_signal}): no change → {score}")
        
        # Sentiment contribution (30%)
        sentiment_score = sentiment.get('overall_score', 0)
        sentiment_contribution = sentiment_score * 0.3
        score += sentiment_contribution
        print(f"Sentiment {sentiment_score} * 0.3: +{sentiment_contribution} → {score}")
        
        # Risk-adjusted contribution (20%) - FIXED to match actual logic
        sharpe_ratio = risk_metrics.get('sharpe_ratio')
        if sharpe_ratio is not None:
            if sharpe_ratio > 1:
                score += 0.1
                print(f"Sharpe {sharpe_ratio} > 1: +0.1 → {score}")
            elif sharpe_ratio < 0:
                score -= 0.1
                print(f"Sharpe {sharpe_ratio} < 0: -0.1 → {score}")
            else:
                print(f"Sharpe {sharpe_ratio} (neutral): no change → {score}")
        else:
            print(f"Sharpe missing: no change → {score}")
        
        # Volatility adjustment (10%) - FIXED to match actual logic
        volatility = risk_metrics.get('volatility_annual')
        if volatility is not None:
            if volatility > 50:
                score -= 0.1
                print(f"Volatility {volatility} > 50: -0.1 → {score}")
            elif volatility < 20:
                score += 0.05
                print(f"Volatility {volatility} < 20: +0.05 → {score}")
            else:
                print(f"Volatility {volatility} (neutral): no change → {score}")
        else:
            print(f"Volatility missing: no change → {score}")
        
        # Final score
        final_score = max(0, min(1, score))
        print(f"\nFinal score (clamped 0-1): {final_score}")
        print(f"Signal strength percentage: {final_score * 100:.1f}%")
        
        # Determine signal label
        if final_score > 0.8:
            signal_label = "STRONG BUY"
        elif final_score > 0.6:
            signal_label = "BUY"
        elif final_score < 0.2:
            signal_label = "STRONG SELL"
        elif final_score < 0.4:
            signal_label = "SELL"
        else:
            signal_label = "HOLD"
        
        print(f"Signal label: {signal_label}")
        
        # Test the actual method
        actual_strength = telegram_handler._calculate_signal_strength(indicators, sentiment, risk_metrics)
        print(f"\nActual method result: {actual_strength}")
        print(f"Actual percentage: {actual_strength * 100:.1f}%")
        
        if abs(actual_strength - final_score) > 0.001:
            print(f"⚠️ WARNING: Manual calculation differs from method result!")
        
    except Exception as e:
        logger.error(f"Error in debug: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_signal_strength())