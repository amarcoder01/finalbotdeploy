import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

qlib_data_dir = os.path.expanduser('~/.qlib/qlib_data/us_data')
os.makedirs(qlib_data_dir, exist_ok=True)

print('Creating minimal Qlib US data structure for testing...')

# Create calendar.txt (trading days)
start_date = datetime(2020, 1, 1)
end_date = datetime(2024, 12, 31)
dates = pd.date_range(start=start_date, end=end_date, freq='D')
trading_days = [d.strftime('%Y-%m-%d') for d in dates if d.weekday() < 5]  # Monday-Friday

with open(os.path.join(qlib_data_dir, 'calendar.txt'), 'w') as f:
    f.write('\n'.join(trading_days))

# Create instruments.csv (stock symbols)
symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'META', 'NVDA', 'NFLX', 'AMD', 'INTC']
instruments_data = []
for symbol in symbols:
    instruments_data.append({
        'instrument': symbol,
        'start_time': '2020-01-01',
        'end_time': '2024-12-31'
    })

instruments_df = pd.DataFrame(instruments_data)
instruments_df.to_csv(os.path.join(qlib_data_dir, 'instruments.csv'), index=False)

# Create features directory and sample data
features_dir = os.path.join(qlib_data_dir, 'features')
os.makedirs(features_dir, exist_ok=True)

# Create sample feature data for each symbol
for symbol in symbols:
    symbol_dir = os.path.join(features_dir, symbol)
    os.makedirs(symbol_dir, exist_ok=True)
    
    # Create sample OHLCV data
    dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
    dates = [d for d in dates if d.weekday() < 5]  # Only trading days
    
    data = []
    base_price = 100.0
    for i, date in enumerate(dates[:100]):  # Limit to 100 days for testing
        price_change = np.random.normal(0, 0.02)  # 2% daily volatility
        base_price *= (1 + price_change)
        
        data.append({
            'datetime': date.strftime('%Y-%m-%d'),
            'open': base_price * (1 + np.random.normal(0, 0.005)),
            'high': base_price * (1 + abs(np.random.normal(0, 0.01))),
            'low': base_price * (1 - abs(np.random.normal(0, 0.01))),
            'close': base_price,
            'volume': int(np.random.uniform(1000000, 10000000))
        })
    
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(symbol_dir, 'day.csv'), index=False)

print(f'Created minimal Qlib US data structure at {qlib_data_dir}')
print(f'Includes {len(symbols)} symbols: {", ".join(symbols)}')
print('Ready for Qlib integration testing!') 