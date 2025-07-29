#!/usr/bin/env python3

import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TradeAiCompanion'))
from advanced_qlib_strategies import AdvancedQlibStrategies
from logger import logger

def portfolio_optimize(symbols_str):
    try:
        # Parse symbols
        symbols = [s.strip().upper() for s in symbols_str.split(',')]
        
        # Initialize strategies
        strategies = AdvancedQlibStrategies()
        
        # Run portfolio optimization
        result = strategies.portfolio_optimization(symbols, 'moderate')
        
        # Format output
        if 'error' in result:
            print(f"❌ Portfolio Optimization Failed\n\nReason(s):\n• {result['error']}")
            return
            
        weights = result.get('weights', {})
        metrics = result.get('metrics', {})
        
        # Print results
        print("\n📊 Portfolio Optimization Results")
        print("\n🎯 Optimized Weights:")
        for symbol, weight in weights.items():
            print(f"• {symbol}: {weight:.2%}")
            
        print("\n📈 Portfolio Metrics:")
        print(f"• Expected Return: {metrics.get('expected_return', 0):.2%}")
        print(f"• Volatility: {metrics.get('volatility', 0):.2%}")
        print(f"• Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        
        if 'warning' in result and result['warning']:
            print(f"\n⚠️ Warning: {result['warning']}")
            
    except Exception as e:
        logger.error(f"Portfolio optimization failed: {e}")
        print("❌ Portfolio Optimization Failed\n")
        print(f"Reason(s):\n• {str(e)}\n")
        print("What you can do:")
        print("• Check that all stock symbols are valid and available.")
        print("• Ensure you have a stable internet connection.")
        print("• Try again later or with different symbols.\n")
        print("If the problem persists, contact support with this message.")

def main():
    if len(sys.argv) != 2:
        print("Usage: /portfolio_optimize SYMBOL1,SYMBOL2,SYMBOL3")
        sys.exit(1)
    portfolio_optimize(sys.argv[1])

if __name__ == "__main__":
    main()