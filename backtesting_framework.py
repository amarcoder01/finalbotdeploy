"""
Backtesting Framework for Trading Strategies
Strategy validation, performance analysis, and optimization
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    sns = None
    SEABORN_AVAILABLE = False
    print("Warning: seaborn not available, using matplotlib only for plotting")
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from logger import logger

@dataclass
class Trade:
    """Trade data structure"""
    entry_date: datetime
    exit_date: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    side: str  # 'long' or 'short'
    symbol: str = 'STOCK'  # Add symbol attribute
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    status: str = 'open'  # 'open', 'closed', 'stopped'

@dataclass
class BacktestResult:
    """Backtest result data structure"""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    trades: List[Trade]
    equity_curve: pd.Series
    drawdown_curve: pd.Series

class BacktestingFramework:
    """Comprehensive backtesting framework"""
    
    def __init__(self, initial_capital: float = 100000, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.current_capital = initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.drawdown_curve = []
        
    def run_backtest(self, data: pd.DataFrame, strategy: Callable, 
                    strategy_params: Dict = None) -> BacktestResult:
        """Run backtest with given strategy"""
        try:
            # Reset state
            self.current_capital = self.initial_capital
            self.positions = {}
            self.trades = []
            self.equity_curve = []
            self.drawdown_curve = []
            
            # Prepare data
            data = data.copy()
            data['Date'] = pd.to_datetime(data.index, utc=True)
            
            # Add technical indicators
            data = self._add_technical_indicators(data)
            
            # Run strategy
            for i in range(len(data)):
                current_data = data.iloc[:i+1]
                if len(current_data) < 20:  # Need minimum data for indicators
                    continue
                
                # Get strategy signal
                signal = strategy(current_data, strategy_params or {})
                
                # Execute signal
                self._execute_signal(signal, current_data.iloc[-1])
                
                # Update equity curve
                self._update_equity_curve(current_data.iloc[-1])
            
            # Close any remaining positions
            self._close_all_positions(data.iloc[-1])
            
            # Calculate results
            return self._calculate_results(data)
            
        except Exception as e:
            logger.error(f"Error in backtest: {e}")
            return self._empty_result()
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to data"""
        try:
            # Moving averages
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            data['EMA_12'] = data['Close'].ewm(span=12).mean()
            data['EMA_26'] = data['Close'].ewm(span=26).mean()
            
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            data['MACD'] = data['EMA_12'] - data['EMA_26']
            data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
            data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
            
            # Bollinger Bands
            data['BB_Middle'] = data['Close'].rolling(window=20).mean()
            bb_std = data['Close'].rolling(window=20).std()
            data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
            data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
            
            # Volume indicators
            data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
            data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
            
            return data
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            return data
    
    def _execute_signal(self, signal: Dict, current_bar: pd.Series):
        """Execute trading signal"""
        try:
            if not signal or 'action' not in signal:
                return
            
            action = signal['action']
            symbol = signal.get('symbol', 'STOCK')
            quantity = signal.get('quantity', 100)
            price = current_bar['Close']
            
            if action == 'buy' and symbol not in self.positions:
                # Open long position
                cost = quantity * price * (1 + self.commission)
                if cost <= self.current_capital:
                    self.positions[symbol] = {
                        'quantity': quantity,
                        'entry_price': price,
                        'entry_date': current_bar.name,
                        'side': 'long'
                    }
                    self.current_capital -= cost
                    
                    # Create trade record
                    trade = Trade(
                        entry_date=current_bar.name,
                        exit_date=None,
                        entry_price=price,
                        exit_price=None,
                        quantity=quantity,
                        side='long',
                        symbol=symbol
                    )
                    self.trades.append(trade)
            
            elif action == 'sell' and symbol in self.positions:
                # Close long position
                position = self.positions[symbol]
                if position['side'] == 'long':
                    revenue = position['quantity'] * price * (1 - self.commission)
                    self.current_capital += revenue
                    
                    # Calculate P&L
                    pnl = revenue - (position['quantity'] * position['entry_price'] * (1 + self.commission))
                    pnl_percent = (pnl / (position['quantity'] * position['entry_price'])) * 100
                    
                    # Update trade record
                    for trade in self.trades:
                        if (trade.symbol == symbol and trade.status == 'open' and 
                            trade.side == 'long'):
                            trade.exit_date = current_bar.name
                            trade.exit_price = price
                            trade.pnl = pnl
                            trade.pnl_percent = pnl_percent
                            trade.status = 'closed'
                            break
                    
                    del self.positions[symbol]
            
            elif action == 'short' and symbol not in self.positions:
                # Open short position
                margin_required = quantity * price * 0.5  # 50% margin requirement
                if margin_required <= self.current_capital:
                    self.positions[symbol] = {
                        'quantity': quantity,
                        'entry_price': price,
                        'entry_date': current_bar.name,
                        'side': 'short'
                    }
                    self.current_capital -= margin_required
                    
                    # Create trade record
                    trade = Trade(
                        entry_date=current_bar.name,
                        exit_date=None,
                        entry_price=price,
                        exit_price=None,
                        quantity=quantity,
                        side='short',
                        symbol=symbol
                    )
                    self.trades.append(trade)
            
            elif action == 'cover' and symbol in self.positions:
                # Close short position
                position = self.positions[symbol]
                if position['side'] == 'short':
                    cost = position['quantity'] * price * (1 + self.commission)
                    margin_returned = position['quantity'] * position['entry_price'] * 0.5
                    self.current_capital += margin_returned - cost
                    
                    # Calculate P&L
                    pnl = (position['quantity'] * position['entry_price']) - cost
                    pnl_percent = (pnl / (position['quantity'] * position['entry_price'])) * 100
                    
                    # Update trade record
                    for trade in self.trades:
                        if (trade.symbol == symbol and trade.status == 'open' and 
                            trade.side == 'short'):
                            trade.exit_date = current_bar.name
                            trade.exit_price = price
                            trade.pnl = pnl
                            trade.pnl_percent = pnl_percent
                            trade.status = 'closed'
                            break
                    
                    del self.positions[symbol]
                    
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
    
    def _update_equity_curve(self, current_bar: pd.Series):
        """Update equity curve"""
        try:
            # Calculate current portfolio value
            portfolio_value = self.current_capital
            
            for symbol, position in self.positions.items():
                if position['side'] == 'long':
                    portfolio_value += position['quantity'] * current_bar['Close']
                elif position['side'] == 'short':
                    # For short positions, we need to account for the liability
                    portfolio_value += position['quantity'] * (position['entry_price'] - current_bar['Close'])
            
            self.equity_curve.append(portfolio_value)
            
        except Exception as e:
            logger.error(f"Error updating equity curve: {e}")
    
    def _close_all_positions(self, current_bar: pd.Series):
        """Close all remaining positions"""
        try:
            for symbol in list(self.positions.keys()):
                position = self.positions[symbol]
                
                if position['side'] == 'long':
                    signal = {'action': 'sell', 'symbol': symbol}
                else:
                    signal = {'action': 'cover', 'symbol': symbol}
                
                self._execute_signal(signal, current_bar)
                
        except Exception as e:
            logger.error(f"Error closing positions: {e}")
    
    def _calculate_results(self, data: pd.DataFrame) -> BacktestResult:
        """Calculate backtest results"""
        try:
            if not self.equity_curve:
                return self._empty_result()
            
            equity_series = pd.Series(self.equity_curve, index=data.index[-len(self.equity_curve):])
            
            # Calculate returns
            returns = equity_series.pct_change().dropna()
            
            # Basic metrics
            total_return = (equity_series.iloc[-1] - self.initial_capital) / self.initial_capital
            annualized_return = self._calculate_annualized_return(equity_series)
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            max_drawdown = self._calculate_max_drawdown(equity_series)
            
            # Trade metrics
            closed_trades = [t for t in self.trades if t.status == 'closed']
            total_trades = len(closed_trades)
            
            if total_trades == 0:
                return self._empty_result()
            
            winning_trades = [t for t in closed_trades if t.pnl > 0]
            losing_trades = [t for t in closed_trades if t.pnl < 0]
            
            win_rate = len(winning_trades) / total_trades
            
            avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
            
            if losing_trades and avg_loss != 0:
                profit_factor = abs(avg_win * len(winning_trades) / (avg_loss * len(losing_trades)))
            else:
                profit_factor = float('inf') if winning_trades else 0
            
            # Consecutive wins/losses
            max_consecutive_wins = self._calculate_max_consecutive(closed_trades, 'win')
            max_consecutive_losses = self._calculate_max_consecutive(closed_trades, 'loss')
            
            # Drawdown curve
            drawdown_curve = self._calculate_drawdown_curve(equity_series)
            
            return BacktestResult(
                total_return=total_return,
                annualized_return=annualized_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor,
                total_trades=total_trades,
                winning_trades=len(winning_trades),
                losing_trades=len(losing_trades),
                avg_win=avg_win,
                avg_loss=avg_loss,
                max_consecutive_wins=max_consecutive_wins,
                max_consecutive_losses=max_consecutive_losses,
                trades=closed_trades,
                equity_curve=equity_series,
                drawdown_curve=drawdown_curve
            )
            
        except Exception as e:
            logger.error(f"Error calculating results: {e}")
            return self._empty_result()
    
    def _calculate_annualized_return(self, equity_series: pd.Series) -> float:
        """Calculate annualized return"""
        try:
            total_days = (equity_series.index[-1] - equity_series.index[0]).days
            if total_days == 0:
                return 0
            
            total_return = (equity_series.iloc[-1] - equity_series.iloc[0]) / equity_series.iloc[0]
            annualized_return = ((1 + total_return) ** (365 / total_days)) - 1
            return annualized_return
            
        except Exception as e:
            logger.error(f"Error calculating annualized return: {e}")
            return 0
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        try:
            if len(returns) == 0:
                return 0
            
            excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
            if excess_returns.std() == 0:
                return 0
            
            sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
            return sharpe_ratio
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0
    
    def _calculate_max_drawdown(self, equity_series: pd.Series) -> float:
        """Calculate maximum drawdown"""
        try:
            peak = equity_series.expanding().max()
            drawdown = (equity_series - peak) / peak
            max_drawdown = drawdown.min()
            return abs(max_drawdown)
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0
    
    def _calculate_drawdown_curve(self, equity_series: pd.Series) -> pd.Series:
        """Calculate drawdown curve"""
        try:
            peak = equity_series.expanding().max()
            drawdown = (equity_series - peak) / peak
            return drawdown
            
        except Exception as e:
            logger.error(f"Error calculating drawdown curve: {e}")
            return pd.Series()
    
    def _calculate_max_consecutive(self, trades: List[Trade], trade_type: str) -> int:
        """Calculate maximum consecutive wins or losses"""
        try:
            max_consecutive = 0
            current_consecutive = 0
            
            for trade in trades:
                if trade_type == 'win' and trade.pnl > 0:
                    current_consecutive += 1
                elif trade_type == 'loss' and trade.pnl < 0:
                    current_consecutive += 1
                else:
                    max_consecutive = max(max_consecutive, current_consecutive)
                    current_consecutive = 0
            
            max_consecutive = max(max_consecutive, current_consecutive)
            return max_consecutive
            
        except Exception as e:
            logger.error(f"Error calculating consecutive {trade_type}: {e}")
            return 0
    
    def _empty_result(self) -> BacktestResult:
        """Return empty backtest result"""
        return BacktestResult(
            total_return=0,
            annualized_return=0,
            sharpe_ratio=0,
            max_drawdown=0,
            win_rate=0,
            profit_factor=0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            avg_win=0,
            avg_loss=0,
            max_consecutive_wins=0,
            max_consecutive_losses=0,
            trades=[],
            equity_curve=pd.Series(),
            drawdown_curve=pd.Series()
        )
    
    def generate_report(self, result: BacktestResult, strategy_name: str = "Strategy") -> str:
        """Generate backtest report"""
        try:
            # Calculate win/loss ratio safely
            if result.avg_loss != 0:
                win_loss_ratio = f"{(result.avg_win/abs(result.avg_loss)):.2f}"
            else:
                win_loss_ratio = "N/A"
            
            report = f"""
📊 **Backtest Report: {strategy_name}**

💰 **Performance Metrics:**
• Total Return: {result.total_return:.2%}
• Annualized Return: {result.annualized_return:.2%}
• Sharpe Ratio: {result.sharpe_ratio:.2f}
• Maximum Drawdown: {result.max_drawdown:.2%}

📈 **Trading Statistics:**
• Total Trades: {result.total_trades}
• Win Rate: {result.win_rate:.2%}
• Profit Factor: {result.profit_factor:.2f}
• Winning Trades: {result.winning_trades}
• Losing Trades: {result.losing_trades}

💵 **Trade Analysis:**
• Average Win: ${result.avg_win:.2f}
• Average Loss: ${result.avg_loss:.2f}
• Max Consecutive Wins: {result.max_consecutive_wins}
• Max Consecutive Losses: {result.max_consecutive_losses}

🎯 **Risk Assessment:**
• Risk-Adjusted Return: {result.sharpe_ratio:.2f}
• Maximum Risk: {result.max_drawdown:.2%}
• Win/Loss Ratio: {win_loss_ratio}
"""
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return "Error generating backtest report"
    
    def plot_results(self, result: BacktestResult, strategy_name: str = "Strategy"):
        """Plot backtest results"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Equity curve
            axes[0, 0].plot(result.equity_curve.index, result.equity_curve.values)
            axes[0, 0].set_title(f'{strategy_name} - Equity Curve')
            axes[0, 0].set_ylabel('Portfolio Value ($)')
            axes[0, 0].grid(True)
            
            # Drawdown curve
            axes[0, 1].fill_between(result.drawdown_curve.index, result.drawdown_curve.values, 0, alpha=0.3, color='red')
            axes[0, 1].set_title(f'{strategy_name} - Drawdown')
            axes[0, 1].set_ylabel('Drawdown (%)')
            axes[0, 1].grid(True)
            
            # Trade distribution
            if result.trades:
                pnls = [t.pnl for t in result.trades]
                axes[1, 0].hist(pnls, bins=20, alpha=0.7)
                axes[1, 0].set_title(f'{strategy_name} - Trade P&L Distribution')
                axes[1, 0].set_xlabel('P&L ($)')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].grid(True)
            
            # Performance metrics
            metrics = ['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate']
            values = [result.total_return, result.sharpe_ratio, result.max_drawdown, result.win_rate]
            axes[1, 1].bar(metrics, values)
            axes[1, 1].set_title(f'{strategy_name} - Key Metrics')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting results: {e}")
            return None

# Predefined strategies
def sma_crossover_strategy(data: pd.DataFrame, params: Dict = None) -> Dict:
    """Simple Moving Average Crossover Strategy"""
    try:
        if len(data) < 50:
            return {}
        
        current = data.iloc[-1]
        prev = data.iloc[-2]
        
        # Check for crossover
        if (prev['SMA_20'] <= prev['SMA_50'] and current['SMA_20'] > current['SMA_50']):
            return {'action': 'buy', 'symbol': 'STOCK', 'quantity': 100}
        elif (prev['SMA_20'] >= prev['SMA_50'] and current['SMA_20'] < current['SMA_50']):
            return {'action': 'sell', 'symbol': 'STOCK', 'quantity': 100}
        
        return {}
        
    except Exception as e:
        logger.error(f"Error in SMA crossover strategy: {e}")
        return {}

def rsi_strategy(data: pd.DataFrame, params: Dict = None) -> Dict:
    """RSI-based Strategy"""
    try:
        if len(data) < 20:
            return {}
        
        current = data.iloc[-1]
        rsi = current['RSI']
        
        # RSI oversold/overbought signals
        if rsi < 30:
            return {'action': 'buy', 'symbol': 'STOCK', 'quantity': 100}
        elif rsi > 70:
            return {'action': 'sell', 'symbol': 'STOCK', 'quantity': 100}
        
        return {}
        
    except Exception as e:
        logger.error(f"Error in RSI strategy: {e}")
        return {}

def macd_strategy(data: pd.DataFrame, params: Dict = None) -> Dict:
    """MACD-based Strategy"""
    try:
        if len(data) < 30:
            return {}
        
        current = data.iloc[-1]
        prev = data.iloc[-2]
        
        # MACD crossover signals
        if (prev['MACD'] <= prev['MACD_Signal'] and current['MACD'] > current['MACD_Signal']):
            return {'action': 'buy', 'symbol': 'STOCK', 'quantity': 100}
        elif (prev['MACD'] >= prev['MACD_Signal'] and current['MACD'] < current['MACD_Signal']):
            return {'action': 'sell', 'symbol': 'STOCK', 'quantity': 100}
        
        return {}
        
    except Exception as e:
        logger.error(f"Error in MACD strategy: {e}")
        return {}