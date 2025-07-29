#!/usr/bin/env python3
"""
Debug script to test the exact advanced_analysis command flow
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import yfinance as yf
import pandas as pd
from enhanced_technical_indicators import EnhancedTechnicalIndicators
from datetime import datetime
import traceback
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def simulate_advanced_analysis_command(symbol="AAPL"):
    """Simulate the exact flow of advanced_analysis_command"""
    print(f"\n🔍 **Simulating Advanced Analysis Command for {symbol}**\n")
    
    try:
        # Step 1: Initialize technical indicators (like in TelegramHandler.__init__)
        print("1. Initializing technical indicators...")
        technical_indicators = EnhancedTechnicalIndicators()
        print("   ✅ Technical indicators initialized")
        
        # Step 2: Get current price data using yfinance (exact same code)
        print("\n2. Fetching current price data...")
        ticker = yf.Ticker(symbol)
        
        try:
            info = ticker.info
            hist = ticker.history(period='1d', interval='1m')
            
            if hist.empty:
                print(f"   ❌ No data available for {symbol}")
                return False
            
            current_price = hist['Close'].iloc[-1]
            open_price = hist['Open'].iloc[0]
            high_price = hist['High'].max()
            low_price = hist['Low'].min()
            volume = hist['Volume'].sum()
            
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
            
            print(f"   ✅ Current price data fetched: ${current_price:.2f}")
            
        except Exception as e:
            print(f"   ❌ Error fetching current data: {e}")
            return False
        
        # Step 3: Get historical data for technical indicators (exact same code)
        print("\n3. Fetching historical data for technical indicators...")
        try:
            hist_data = ticker.history(period='3mo')
            if not hist_data.empty:
                print(f"   ✅ Historical data fetched: {len(hist_data)} records")
                print(f"   - Columns: {list(hist_data.columns)}")
                print(f"   - Index type: {type(hist_data.index)}")
                print(f"   - Data types: {hist_data.dtypes.to_dict()}")
                
                # This is the critical line that might be failing
                print("\n4. Calculating technical indicators...")
                indicators = technical_indicators.calculate_all_indicators(hist_data)
                
                if not indicators:
                    print("   ❌ No indicators calculated")
                    return False
                
                print(f"   ✅ Indicators calculated: {len(indicators)} total")
                market_data['technical_indicators'] = indicators
                market_data['historical_data'] = hist_data.tail(30).to_dict('records')
                
            else:
                print("   ⚠️ No historical data available")
                market_data['technical_indicators'] = {}
                
        except Exception as e:
            print(f"   ❌ Error fetching historical data: {e}")
            print(f"   Traceback: {traceback.format_exc()}")
            market_data['technical_indicators'] = {}
        
        # Step 4: Format response (exact same code)
        print("\n5. Formatting response...")
        try:
            response = f"""
🔍 **Advanced Analysis: {symbol}**

📊 **Price Data:**
• Current Price: ${market_data.get('price', 0):.2f}
• Change: ${market_data.get('change', 0):+.2f} ({market_data.get('change_percent', 0):+.2f}%)
• Volume: {market_data.get('volume', 0):,.0f}
• High: ${market_data.get('high', 0):.2f}
• Low: ${market_data.get('low', 0):.2f}
• Open: ${market_data.get('open', 0):.2f}
"""
            
            # Add technical indicators if available
            if 'technical_indicators' in market_data and market_data['technical_indicators']:
                response += "\n📈 **Technical Indicators:**\n"
                tech = market_data['technical_indicators']
                
                print(f"   - Available indicators: {list(tech.keys())[:10]}...")
                
                # Key indicators
                if 'rsi' in tech and not pd.isna(tech['rsi']):
                    rsi_signal = "Oversold" if tech['rsi'] < 30 else "Overbought" if tech['rsi'] > 70 else "Neutral"
                    response += f"• RSI: {tech['rsi']:.1f} ({rsi_signal})\n"
                    print(f"   ✅ RSI: {tech['rsi']:.1f}")
                else:
                    print(f"   ⚠️ RSI not available or NaN: {tech.get('rsi', 'Missing')}")
                
                if 'macd' in tech and not pd.isna(tech['macd']):
                    response += f"• MACD: {tech['macd']:.4f}\n"
                    print(f"   ✅ MACD: {tech['macd']:.4f}")
                else:
                    print(f"   ⚠️ MACD not available or NaN: {tech.get('macd', 'Missing')}")
                
                if 'sma_20' in tech and 'sma_50' in tech and not pd.isna(tech['sma_20']) and not pd.isna(tech['sma_50']):
                    trend = "Bullish" if tech['sma_20'] > tech['sma_50'] else "Bearish"
                    response += f"• SMA 20/50: ${tech['sma_20']:.2f} / ${tech['sma_50']:.2f} ({trend})\n"
                    print(f"   ✅ SMA 20/50: ${tech['sma_20']:.2f} / ${tech['sma_50']:.2f}")
                else:
                    print(f"   ⚠️ SMA not available: SMA20={tech.get('sma_20', 'Missing')}, SMA50={tech.get('sma_50', 'Missing')}")
                
                if 'bb_upper' in tech and 'bb_lower' in tech and not pd.isna(tech['bb_upper']) and not pd.isna(tech['bb_lower']):
                    current_price = market_data.get('price', 0)
                    if current_price > tech['bb_upper']:
                        bb_signal = "Above Upper Band"
                    elif current_price < tech['bb_lower']:
                        bb_signal = "Below Lower Band"
                    else:
                        bb_signal = "Within Bands"
                    response += f"• Bollinger Bands: ${tech['bb_lower']:.2f} - ${tech['bb_upper']:.2f} ({bb_signal})\n"
                    print(f"   ✅ Bollinger Bands: ${tech['bb_lower']:.2f} - ${tech['bb_upper']:.2f}")
                else:
                    print(f"   ⚠️ Bollinger Bands not available: Upper={tech.get('bb_upper', 'Missing')}, Lower={tech.get('bb_lower', 'Missing')}")
                
                # Add volume analysis
                if 'volume_ratio' in tech and not pd.isna(tech['volume_ratio']):
                    vol_signal = "High" if tech['volume_ratio'] > 1.5 else "Low" if tech['volume_ratio'] < 0.5 else "Normal"
                    response += f"• Volume: {vol_signal} ({tech['volume_ratio']:.1f}x avg)\n"
                    print(f"   ✅ Volume ratio: {tech['volume_ratio']:.1f}x")
                else:
                    print(f"   ⚠️ Volume ratio not available: {tech.get('volume_ratio', 'Missing')}")
                
                # Signals from enhanced technical indicators
                if hasattr(tech, 'get') and 'signals' in tech:
                    signals = tech['signals']
                    response += f"\n🎯 **Trading Signals:** {signals.get('overall_signal', 'NEUTRAL')}\n"
                    if signals.get('buy_signals'):
                        response += f"• Buy: {', '.join(signals['buy_signals'])}\n"
                    if signals.get('sell_signals'):
                        response += f"• Sell: {', '.join(signals['sell_signals'])}\n"
                    print(f"   ✅ Signals: {signals.get('overall_signal', 'NEUTRAL')}")
                else:
                    print(f"   ⚠️ No signals available in tech indicators")
            else:
                response += "\n📈 **Technical Indicators:** Not available (insufficient historical data)\n"
                print("   ⚠️ No technical indicators available")
            
            response += f"\n**Data Source:** REAL-TIME MARKET DATA ✅"
            response += f"\n**Analysis Time:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
            
            print("   ✅ Response formatted successfully")
            print(f"\n📄 **Final Response Preview (first 500 chars):**\n{response[:500]}...")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error formatting response: {e}")
            print(f"   Traceback: {traceback.format_exc()}")
            return False
        
    except Exception as e:
        print(f"\n❌ **Critical Error in advanced analysis simulation:**")
        print(f"   Error: {str(e)}")
        print(f"   Type: {type(e).__name__}")
        print(f"\n   Full traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 **Advanced Analysis Command Debug Test**")
    print("=" * 60)
    
    # Test with a few different symbols
    test_symbols = ["AAPL", "TSLA"]
    
    for i, symbol in enumerate(test_symbols, 1):
        print(f"\n\n🧪 **Test {i}/{len(test_symbols)}: {symbol}**")
        print("-" * 40)
        
        success = simulate_advanced_analysis_command(symbol)
        
        if success:
            print(f"\n✅ **Test {i} PASSED for {symbol}**")
        else:
            print(f"\n❌ **Test {i} FAILED for {symbol}**")
            print("   This indicates a real issue that needs fixing!")
    
    print(f"\n\n{'='*60}")
    print("🏁 **Debug test completed!**")
    print("\nIf any valid symbols failed, the issue is identified above.")