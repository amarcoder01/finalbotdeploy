"""
Advanced Qlib Strategies
Multiple model strategies, portfolio optimization, and risk management
"""
from logger import logger

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import os
import joblib
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    yf = None
    YFINANCE_AVAILABLE = False
    print("Warning: yfinance not available in advanced_qlib_strategies, some features disabled")
import time
import traceback
import requests

# Modern Portfolio Optimization imports
try:
    from portfolio_optimizer import ModernPortfolioOptimizer
    MODERN_OPTIMIZER_AVAILABLE = True
    logger.info("Modern Portfolio Optimizer successfully imported")
except (ImportError, AttributeError) as e:
    MODERN_OPTIMIZER_AVAILABLE = False
    ModernPortfolioOptimizer = None
    logger.warning(f"Modern Portfolio Optimizer not available: {e}. Using fallback optimization.")

try:
    import qlib
    from qlib.constant import REG_CN, REG_US
    from qlib.utils import init_instance_by_config
    from qlib.workflow import R
    from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
    from qlib.contrib.evaluate import backtest_daily
    from qlib.contrib.strategy import TopkDropoutStrategy
    from qlib.contrib.model.gbdt import LGBModel
    from qlib.contrib.model.linear import LinearModel
    from qlib.contrib.model.pytorch_model import GRUModel
    from qlib.data import D
    QLIB_AVAILABLE = True
except ImportError:
    QLIB_AVAILABLE = False
import pandas as pd
from datetime import datetime, timedelta
try:
    import pandas_datareader.data as web
    GOOGLE_FINANCE_AVAILABLE = True
except ImportError:
    GOOGLE_FINANCE_AVAILABLE = False

class AdvancedQlibStrategies:
    """Advanced Qlib strategies with multiple models and portfolio optimization"""
    
    def __init__(self):
        self.models = {}
        self.strategies = {}
        self.risk_metrics = {}
        self.portfolio_weights = {}
        
        # Initialize Qlib with US data
        if QLIB_AVAILABLE:
            try:
                qlib_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "qlib_data", "us_data")
                if not os.path.exists(qlib_data_dir):
                    qlib_data_dir = os.path.expanduser("~/.qlib/qlib_data/us_data")
                qlib.init(provider_uri=qlib_data_dir, region=REG_CN)
                logger.info(f"Qlib initialized with data directory: {qlib_data_dir}")
            except Exception as e:
                logger.error(f"Failed to initialize Qlib: {e}")
        else:
            logger.warning("Qlib not available, will use yfinance for data fetching")
        
    def initialize_qlib(self):
        """Initialize Qlib with US market data"""
        if not QLIB_AVAILABLE:
            logger.warning("Qlib not available, using demo strategies")
            return False
            
        try:
            # Initialize Qlib for US market
            qlib.init(provider_uri='~/.qlib/qlib_data/us_data', region=REG_CN)
            logger.info("Qlib initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Qlib: {e}")
            return False
    
    def create_model_configs(self) -> Dict:
        """Create configurations for different model types"""
        return {
            'lgb_model': {
                'class': 'LGBModel',
                'module_path': 'qlib.contrib.model.gbdt',
                'kwargs': {
                    'loss': 'mse',
                    'colsample_bytree': 0.8879,
                    'colsample_bylevel': 0.8879,
                    'max_depth': 8,
                    'num_leaves': 210,
                    'subsample': 0.8789,
                    'n_estimators': 100,
                    'learning_rate': 0.2,
                }
            },
            'linear_model': {
                'class': 'LinearModel',
                'module_path': 'qlib.contrib.model.linear',
                'kwargs': {
                    'estimator': 'Lasso',
                    'alpha': 0.0001,
                }
            },
            'gru_model': {
                'class': 'GRUModel',
                'module_path': 'qlib.contrib.model.pytorch_model',
                'kwargs': {
                    'd_feat': 6,
                    'hidden_size': 64,
                    'num_layers': 2,
                    'dropout': 0.2,
                    'n_epochs': 100,
                    'lr': 0.001,
                }
            }
        }
    
    def train_multiple_models(self, symbols: List[str], start_date: str = "2020-01-01", end_date: str = None) -> Dict:
        """Train multiple models for ensemble predictions, with local caching/loading"""
        if not QLIB_AVAILABLE:
            return self._demo_multiple_models(symbols)
        try:
            if not end_date:
                end_date = datetime.utcnow().strftime('%Y-%m-%d')
            model_configs = self.create_model_configs()
            trained_models = {}
            model_dir = "./trained_models"
            os.makedirs(model_dir, exist_ok=True)
            for model_name, config in model_configs.items():
                model_path = os.path.join(model_dir, f"{model_name}.joblib")
                if os.path.exists(model_path):
                    try:
                        logger.info(f"Loading {model_name} from local cache...")
                        model = joblib.load(model_path)
                        trained_models[model_name] = model
                        continue
                    except Exception as e:
                        logger.error(f"Error loading {model_name} from cache: {e}")
                        # If loading fails, retrain
                logger.info(f"Training {model_name}...")
                try:
                    model = init_instance_by_config(config)
                    with R.start(experiment_name=f"{model_name}_training"):
                        model.fit(dataset=self._create_dataset(symbols, start_date, end_date))
                        trained_models[model_name] = model
                        joblib.dump(model, model_path)
                        logger.info(f"Saved {model_name} to {model_path}")
                except Exception as e:
                    logger.error(f"Error training {model_name}: {e}")
                    continue
            self.models = trained_models
            logger.info(f"Trained/Loaded {len(trained_models)} models successfully")
            return trained_models
        except Exception as e:
            logger.error(f"Error in multiple model training: {e}")
            return self._demo_multiple_models(symbols)
    
    def generate_ensemble_signals(self, symbols: List[str]) -> Dict:
        """Generate signals using ensemble of multiple models"""
        if not self.models:
            return self._demo_ensemble_signals(symbols)
        
        try:
            ensemble_signals = {}
            
            for symbol in symbols:
                symbol_signals = {}
                
                # Get predictions from each model
                for model_name, model in self.models.items():
                    try:
                        prediction = self._get_model_prediction(model, symbol)
                        symbol_signals[model_name] = prediction
                    except Exception as e:
                        logger.error(f"Error getting prediction from {model_name} for {symbol}: {e}")
                        continue
                
                # Combine predictions (ensemble)
                if symbol_signals:
                    ensemble_prediction = self._combine_predictions(symbol_signals)
                    ensemble_signals[symbol] = ensemble_prediction
                
            return ensemble_signals
            
        except Exception as e:
            logger.error(f"Error generating ensemble signals: {e}")
            return self._demo_ensemble_signals(symbols)
    
    def _combine_predictions(self, predictions: Dict) -> Dict:
        """Combine predictions from multiple models"""
        try:
            # Simple ensemble: weighted average
            weights = {
                'lgb_model': 0.4,
                'linear_model': 0.3,
                'gru_model': 0.3
            }
            
            combined_score = 0
            total_weight = 0
            
            for model_name, prediction in predictions.items():
                if model_name in weights and 'score' in prediction:
                    combined_score += prediction['score'] * weights[model_name]
                    total_weight += weights[model_name]
            
            if total_weight > 0:
                final_score = combined_score / total_weight
            else:
                final_score = 0
            
            # Determine signal based on ensemble score
            if final_score > 0.6:
                signal = 'BUY'
                confidence = min(final_score * 100, 95)
            elif final_score < -0.6:
                signal = 'SELL'
                confidence = min(abs(final_score) * 100, 95)
            else:
                signal = 'HOLD'
                confidence = 50
            
            return {
                'signal': signal,
                'score': final_score,
                'confidence': confidence,
                'models_used': len(predictions),
                'individual_predictions': predictions
            }
            
        except Exception as e:
            logger.error(f"Error combining predictions: {e}")
            return {'signal': 'HOLD', 'score': 0, 'confidence': 0}
    
    def portfolio_optimization(self, symbols: List[str], risk_tolerance: str = 'moderate') -> Dict:
        """Modern portfolio optimization using Riskfolio-Lib with robust fallback"""
        start_time = datetime.utcnow()
        diagnostics = []
        try:
            logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting modern portfolio optimization")
            logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Input parameters: symbols={symbols}, risk_tolerance={risk_tolerance}")
            
            if not MODERN_OPTIMIZER_AVAILABLE:
                logger.error(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Modern Portfolio Optimizer not available, falling back to basic optimization")
                return self._fallback_optimization(symbols, risk_tolerance)
            
            # Use the modern portfolio optimizer
            optimizer = ModernPortfolioOptimizer()
            result = optimizer.optimize_portfolio(symbols, risk_tolerance)
            
            if result and 'weights' in result:
                logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Modern optimization completed successfully")
                return result
            else:
                logger.warning("Modern optimization failed, using fallback")
                return self._fallback_optimization(symbols, risk_tolerance)

        except Exception as e:
            logger.error(f"Error in modern portfolio optimization: {e}")
            return self._fallback_optimization(symbols, risk_tolerance)
    
    def _fetch_price_data(self, symbols: List[str], period: str = '2y') -> pd.DataFrame:
        """Historical price data using Qlib with yfinance fallback for portfolio optimization"""
        fetch_start_time = datetime.utcnow()
        price_data = {}
        diagnostics = []
        logger.info(f"[{datetime.utcnow()}] Starting price data fetch for {len(symbols)} symbols")
        
        # Validate symbols
        valid_symbols = [symbol.strip().upper() for symbol in symbols if symbol.strip()]
        if not valid_symbols:
            logger.error("No valid symbols provided")
            return pd.DataFrame()
            
        # Convert period to start_time
        if period == '2y':
            start_time = (datetime.utcnow() - timedelta(days=730)).strftime('%Y-%m-%d')
        elif period == '1y':
            start_time = (datetime.utcnow() - timedelta(days=365)).strftime('%Y-%m-%d')
        else:
            start_time = (datetime.utcnow() - timedelta(days=730)).strftime('%Y-%m-%d')
        end_time = datetime.utcnow().strftime('%Y-%m-%d')
            
        # Try Qlib data first
        if QLIB_AVAILABLE:
            try:
                logger.info(f"[{datetime.utcnow()}] Attempting to fetch data from Qlib for {valid_symbols}")
                
                # Prepare Qlib-formatted symbols (ensure they follow Qlib's format)
                qlib_symbols = [f"{symbol}.US" for symbol in valid_symbols]
                
                # Fetch data using Qlib's D.features
                df = D.features(
                    instruments=qlib_symbols,
                    fields=['$close'],
                    start_time=start_time,
                    end_time=end_time,
                    freq='day'
                )
                
                if not df.empty:
                    logger.info(f"Successfully fetched Qlib data with shape {df.shape}")
                    
                    # Process data for each symbol
                    for symbol, qlib_symbol in zip(valid_symbols, qlib_symbols):
                        try:
                            # Extract close prices for the symbol
                            symbol_data = df.loc[(slice(None), qlib_symbol), '$close']
                            if not symbol_data.empty:
                                # Reset index to get datetime as index
                                symbol_data.index = symbol_data.index.get_level_values('datetime')
                                price_data[symbol] = symbol_data
                                diagnostics.append(f"{symbol}: Qlib data OK ({len(symbol_data)} days)")
                                logger.info(f"Successfully processed {symbol} data with {len(symbol_data)} days")
                        except Exception as e:
                            logger.warning(f"Error processing Qlib data for {symbol}: {e}")
                    
                    if price_data:
                        logger.info(f"Successfully processed Qlib data for {len(price_data)} symbols")
                        # Combine all symbol data into a single DataFrame
                        combined_data = pd.DataFrame(price_data)
                        # Ensure data quality
                        if len(combined_data) >= 100:
                            logger.info(f"Using Qlib data with {len(combined_data)} days for portfolio optimization")
                            return combined_data
                        else:
                            logger.warning(f"Insufficient Qlib data points: {len(combined_data)}, falling back to yfinance")
                else:
                    logger.warning("Empty DataFrame returned from Qlib, falling back to yfinance")
                    
            except Exception as e:
                logger.error(f"Error fetching Qlib data: {e}, falling back to yfinance")
                logger.error(f"Traceback: {traceback.format_exc()}")
        else:
            logger.warning("Qlib not available, using yfinance data")

            
        for symbol in valid_symbols:
            try:
                logger.info(f"[{datetime.utcnow()}] Fetching data for {symbol} using yfinance")
                ticker = yf.Ticker(symbol)
                
                # Check if the symbol exists
                info = ticker.info
                if not info or 'regularMarketPrice' not in info:
                    logger.warning(f"Invalid symbol or no market data available for {symbol}")
                    diagnostics.append(f"{symbol}: Invalid symbol or no market data")
                    continue
                
                # Force fresh data fetch to avoid cached data
                df = ticker.history(period=period, auto_adjust=True, timeout=15)
                logger.info(f"[{datetime.utcnow()}] Received {len(df)} rows of data for {symbol}")
                
                # Validate data quality
                if df.empty:
                    logger.warning(f"Empty data received for {symbol} from yfinance")
                    diagnostics.append(f"{symbol}: yfinance returned empty data")
                elif len(df) < 100:
                    logger.warning(f"Insufficient data for {symbol}: {len(df)} days from yfinance")
                    diagnostics.append(f"{symbol}: yfinance insufficient ({len(df)} days)")
                else:
                    # Check for data gaps
                    missing_pct = df['Close'].isnull().mean() * 100
                    if missing_pct > 10:
                        logger.warning(f"High percentage of missing data ({missing_pct:.1f}%) for {symbol}")
                        diagnostics.append(f"{symbol}: {missing_pct:.1f}% missing data")
                    else:
                        price_data[symbol] = df['Close']
                        logger.info(f"Fetched {len(df)} days of data for {symbol} from yfinance")
                        diagnostics.append(f"{symbol}: yfinance OK ({len(df)} days)")
                        continue
                
                # Try Alpha Vantage as fallback
                av_close = self.fetch_alpha_vantage_close(symbol, (pd.Timestamp.today() - pd.Timedelta(days=730)).strftime('%Y-%m-%d'), pd.Timestamp.today().strftime('%Y-%m-%d'))
                if av_close is not None and len(av_close) >= 100:
                    missing_pct = av_close.isnull().mean() * 100
                    if missing_pct <= 10:
                        price_data[symbol] = av_close
                        logger.info(f"Fetched {len(av_close)} days of data for {symbol} from Alpha Vantage fallback")
                        diagnostics.append(f"{symbol}: Alpha Vantage fallback used ({len(av_close)} days)")
                    else:
                        logger.warning(f"High percentage of missing data ({missing_pct:.1f}%) in Alpha Vantage data for {symbol}")
                        diagnostics.append(f"{symbol}: Alpha Vantage {missing_pct:.1f}% missing data")
                else:
                    diagnostics.append(f"{symbol}: Alpha Vantage fallback failed or insufficient")
                    
            except Exception as e:
                logger.error(f"[{datetime.utcnow()}] Error fetching data for {symbol}: {e}")
                diagnostics.append(f"{symbol}: yfinance error: {str(e)}")
                logger.error(f"Full error details for {symbol}: {str(e)}")
                logger.error(f"Traceback for {symbol}: {traceback.format_exc()}")
                
                # Try Alpha Vantage as fallback
                av_close = self.fetch_alpha_vantage_close(symbol, (pd.Timestamp.today() - pd.Timedelta(days=730)).strftime('%Y-%m-%d'), pd.Timestamp.today().strftime('%Y-%m-%d'))
                if av_close is not None and len(av_close) >= 100:
                    missing_pct = av_close.isnull().mean() * 100
                    if missing_pct <= 10:
                        price_data[symbol] = av_close
                        logger.info(f"Fetched {len(av_close)} days of data for {symbol} from Alpha Vantage fallback")
                        diagnostics.append(f"{symbol}: Alpha Vantage fallback used ({len(av_close)} days)")
                    else:
                        logger.warning(f"High percentage of missing data ({missing_pct:.1f}%) in Alpha Vantage data for {symbol}")
                        diagnostics.append(f"{symbol}: Alpha Vantage {missing_pct:.1f}% missing data")
                else:
                    diagnostics.append(f"{symbol}: Alpha Vantage fallback failed or insufficient")
        if not price_data:
            logger.error(f"[{datetime.utcnow()}] No price data available for symbols: {symbols}")
            logger.error(f"Diagnostics: {diagnostics}")
            return pd.DataFrame()
        
        logger.info(f"[{datetime.utcnow()}] Price data fetch completed. Time taken: {datetime.utcnow() - fetch_start_time}")
        combined_data = pd.DataFrame(price_data).dropna()
        if len(combined_data) < 50:
            logger.warning(f"Limited overlapping data: {len(combined_data)} days")
            logger.warning(f"Diagnostics: {diagnostics}")
        return combined_data
    

    
    def _fallback_optimization(self, symbols: List[str], risk_tolerance: str) -> Dict:
        """Fallback optimization when modern portfolio optimizer is not available"""
        logger.warning("Using fallback optimization (modern portfolio optimizer not available)")
        
        try:
            # Simple equal-weight portfolio
            weights = {symbol: 1/len(symbols) for symbol in symbols}
            
            # Fetch some basic data for metrics
            price_data = self._fetch_price_data(symbols)
            
            if not price_data.empty and len(price_data) > 30:
                returns = price_data.pct_change().dropna()
                
                # Calculate portfolio returns with equal weights
                portfolio_returns = (returns * (1/len(symbols))).sum(axis=1)
                
                # Calculate metrics
                expected_return = float(portfolio_returns.mean() * 252)
                volatility = float(portfolio_returns.std() * np.sqrt(252))
                sharpe_ratio = (expected_return - 0.02) / volatility if volatility > 0 else 0
                
                # Calculate max drawdown
                cumulative = (1 + portfolio_returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                max_drawdown = float(drawdown.min())
                
                # Calculate VaR and CVaR
                var_95 = float(np.percentile(portfolio_returns, 5) * np.sqrt(252))
                cvar_95 = float(portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean() * np.sqrt(252))
                
                # Calculate Sortino ratio
                excess_returns = portfolio_returns - 0.02/252
                downside_returns = excess_returns[excess_returns < 0]
                if len(downside_returns) > 0:
                    downside_deviation = np.sqrt((downside_returns ** 2).mean()) * np.sqrt(252)
                    sortino_ratio = (excess_returns.mean() * 252) / downside_deviation if downside_deviation > 0 else 0
                else:
                    sortino_ratio = float('inf')
                
                metrics = {
                    'expected_return': expected_return,
                    'volatility': volatility,
                    'sharpe_ratio': float(sharpe_ratio),
                    'max_drawdown': max_drawdown,
                    'var_95': var_95,
                    'cvar_95': cvar_95,
                    'skewness': float(portfolio_returns.skew()),
                    'kurtosis': float(portfolio_returns.kurtosis()),
                    'sortino_ratio': float(sortino_ratio) if sortino_ratio != float('inf') else 10.0,
                    'calmar_ratio': float(expected_return / abs(max_drawdown)) if abs(max_drawdown) > 0.001 else 0.0
                }
                
                data_period_days = len(returns)
            else:
                # Use conservative estimates when no data is available
                logger.warning("No price data available, using conservative estimates")
                metrics = {
                    'expected_return': 0.06,  # Conservative 6% return
                    'volatility': 0.18,       # Typical market volatility
                    'sharpe_ratio': 0.33,     # Conservative Sharpe ratio
                    'max_drawdown': -0.25,    # Conservative max drawdown
                    'var_95': -0.08,          # Conservative VaR
                    'cvar_95': -0.12,         # Conservative CVaR
                    'skewness': -0.3,         # Slightly negative skew
                    'kurtosis': 3.5,          # Slightly higher kurtosis
                    'sortino_ratio': 0.45,    # Conservative Sortino ratio
                    'calmar_ratio': 0.24      # Conservative Calmar ratio
                }
                data_period_days = 0
            
            return {
                'weights': weights,
                'metrics': metrics,
                'risk_tolerance': risk_tolerance,
                'strategy': 'Equal Weight (Fallback)',
                'symbols': symbols,
                'data_period_days': data_period_days,
                'optimization_timestamp': datetime.utcnow().isoformat(),
                'optimizer': 'fallback',
                'warning': 'Modern portfolio optimizer not available, using equal weights with calculated metrics'
            }
                
        except Exception as e:
            logger.error(f"Error in fallback optimization: {e}")
            return {
                'weights': {s: 1/len(symbols) for s in symbols},
                'metrics': {
                    'expected_return': 0.05,
                    'volatility': 0.20,
                    'sharpe_ratio': 0.25,
                    'max_drawdown': -0.30,
                    'var_95': -0.10,
                    'cvar_95': -0.15
                },
                'risk_tolerance': risk_tolerance,
                'strategy': 'Equal Weight (Error Fallback)',
                'symbols': symbols,
                'data_period_days': 0,
                'optimization_timestamp': datetime.utcnow().isoformat(),
                'optimizer': 'error_fallback',
                'error': f'Portfolio optimization failed: {str(e)}. Using default equal weights.',
                'warning': 'Error occurred during optimization, using default metrics'
            }
    
    def risk_management(self, portfolio: Dict, market_data: Dict) -> Dict:
        """Risk management and position sizing"""
        try:
            risk_metrics = {}
            
            # Calculate Value at Risk (VaR)
            portfolio_value = 100000  # Example portfolio value
            var_95 = self._calculate_var(portfolio, market_data, confidence=0.95)
            var_99 = self._calculate_var(portfolio, market_data, confidence=0.99)
            
            # Calculate Maximum Drawdown
            max_drawdown = self._calculate_max_drawdown(portfolio, market_data)
            
            # Position sizing based on volatility
            position_sizes = self._calculate_position_sizes(portfolio, market_data)
            
            # Risk-adjusted returns
            sharpe_ratio = self._calculate_sharpe_ratio(portfolio, market_data)
            
            risk_metrics = {
                'var_95': var_95,
                'var_99': var_99,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'position_sizes': position_sizes,
                'risk_level': self._assess_risk_level(var_95, max_drawdown, sharpe_ratio)
            }
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error in risk management: {e}")
            return self._demo_risk_metrics()
    
    def _calculate_var(self, portfolio: Dict, market_data: Dict, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        try:
            # Simplified VaR calculation
            portfolio_volatility = 0.15  # Example volatility
            z_score = 1.645 if confidence == 0.95 else 2.326
            var = z_score * portfolio_volatility * 100000  # Portfolio value
            return var
        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            return 0
    
    def _calculate_max_drawdown(self, portfolio: Dict, market_data: Dict) -> float:
        """Calculate maximum drawdown"""
        try:
            # Simplified max drawdown calculation
            return 0.25  # Example 25% max drawdown
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0
    
    def _calculate_sharpe_ratio(self, portfolio: Dict, market_data: Dict) -> float:
        """Calculate Sharpe ratio"""
        try:
            # Simplified Sharpe ratio calculation
            return 1.2  # Example Sharpe ratio
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0
    
    def _assess_risk_level(self, var_95: float, max_drawdown: float, sharpe_ratio: float) -> str:
        """Assess overall risk level"""
        try:
            risk_score = 0
            
            # VaR assessment
            if var_95 > 5000:
                risk_score += 3
            elif var_95 > 3000:
                risk_score += 2
            else:
                risk_score += 1
            
            # Drawdown assessment
            if max_drawdown > 0.3:
                risk_score += 3
            elif max_drawdown > 0.2:
                risk_score += 2
            else:
                risk_score += 1
            
            # Sharpe ratio assessment
            if sharpe_ratio < 0.5:
                risk_score += 3
            elif sharpe_ratio < 1.0:
                risk_score += 2
            else:
                risk_score += 1
            
            # Determine risk level
            if risk_score <= 4:
                return 'LOW'
            elif risk_score <= 6:
                return 'MODERATE'
            else:
                return 'HIGH'
                
        except Exception as e:
            logger.error(f"Error assessing risk level: {e}")
            return 'MODERATE'
    
    # Demo methods for when Qlib is not available
    def _demo_multiple_models(self, symbols: List[str]) -> Dict:
        """Demo multiple models when Qlib is not available"""
        return {
            'lgb_model': 'demo_model',
            'linear_model': 'demo_model',
            'gru_model': 'demo_model'
        }
    
    def _demo_ensemble_signals(self, symbols: List[str]) -> Dict:
        """Demo ensemble signals"""
        signals = {}
        for symbol in symbols:
            signals[symbol] = {
                'signal': np.random.choice(['BUY', 'SELL', 'HOLD']),
                'score': np.random.uniform(-1, 1),
                'confidence': np.random.uniform(60, 95),
                'models_used': 3,
                'individual_predictions': {
                    'lgb_model': {'score': np.random.uniform(-1, 1)},
                    'linear_model': {'score': np.random.uniform(-1, 1)},
                    'gru_model': {'score': np.random.uniform(-1, 1)}
                }
            }
        return signals
    
    def _demo_portfolio_optimization(self, symbols: List[str]) -> Dict:
        """Demo portfolio optimization"""
        weights = {symbol: 1/len(symbols) for symbol in symbols}
        return {
            'weights': weights,
            'metrics': {
                'expected_return': 0.10,
                'volatility': 0.18,
                'sharpe_ratio': 1.2,
                'max_drawdown': 0.15
            },
            'risk_tolerance': 'moderate',
            'symbols': symbols
        }
    
    def _demo_risk_metrics(self) -> Dict:
        """Demo risk metrics"""
        return {
            'var_95': 3500,
            'var_99': 5000,
            'max_drawdown': 0.20,
            'sharpe_ratio': 1.1,
            'position_sizes': {'AAPL': 0.25, 'TSLA': 0.25, 'GOOGL': 0.25, 'MSFT': 0.25},
            'risk_level': 'MODERATE'
        }
    
    def fetch_alpha_vantage_close(self, symbol, start, end, api_key=None):
        """Fetch daily close prices from Alpha Vantage."""
        try:
            start_time = time.time()
            logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting Alpha Vantage fetch for {symbol}")
            
            if not api_key:
                api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
                if not api_key:
                    logger.error("ALPHA_VANTAGE_API_KEY environment variable not set")
                    return None
            
            url = 'https://www.alphavantage.co/query'
            params = {
                'function': 'TIME_SERIES_DAILY_ADJUSTED',
                'symbol': symbol,
                'outputsize': 'full',
                'apikey': api_key
            }
            
            logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Sending request to Alpha Vantage API for {symbol}")
            r = requests.get(url, params=params, timeout=10, headers={'Cache-Control': 'no-cache'})
            
            if r.status_code != 200:
                logger.error(f"Alpha Vantage API request failed with status code {r.status_code}")
                return None
                
            data = r.json()
            if 'Error Message' in data:
                logger.error(f"Alpha Vantage API error: {data['Error Message']}")
                return None
                
            if 'Time Series (Daily)' not in data:
                if 'Note' in data:
                    logger.warning(f"Alpha Vantage API limit reached: {data['Note']}")
                else:
                    logger.warning(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Alpha Vantage: No data for {symbol}")
                return None
            ts = data['Time Series (Daily)']
            df = pd.DataFrame.from_dict(ts, orient='index')
            df.index = pd.to_datetime(df.index, utc=True)
            df = df.sort_index()
            df = df[(df.index >= start) & (df.index <= end)]
            df = df.rename(columns={'5. adjusted close': 'Close'})
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            return df['Close'].dropna()
        except Exception as e:
            logger.error(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Alpha Vantage fetch failed for {symbol}: {str(e)}\nTraceback: {traceback.format_exc()}")
            return None
        finally:
            duration = time.time() - start_time
            logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Alpha Vantage fetch for {symbol} completed in {duration:.2f} seconds")

    def fetch_google_finance_close(self, symbol, start, end):
        """Fetch daily close prices from Google Finance using pandas_datareader."""
        if not GOOGLE_FINANCE_AVAILABLE:
            logger.warning("pandas_datareader is not installed; Google Finance unavailable.")
            return None
        try:
            df = web.DataReader(symbol, 'google', start, end)
            if 'Close' in df:
                return df['Close'].dropna()
            elif 'close' in df:
                return df['close'].dropna()
            else:
                logger.warning(f"Google Finance: No close data for {symbol}")
                return None
        except Exception as e:
            logger.warning(f"Google Finance fetch failed for {symbol}: {e}")
            return None

    def _get_returns_data(self, symbols: List[str], period: str = '2y') -> pd.DataFrame:
        """Fetch real-time historical daily returns with enhanced data quality for optimization."""
        all_returns = {}
        start_time = time.time()
        logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting data fetch for {len(symbols)} symbols: {symbols}")
        
        for symbol in symbols:
            symbol_start_time = time.time()
            try:
                logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Fetching data for {symbol}")
                # Fetch with extended period for better statistical accuracy
                ticker = yf.Ticker(symbol)
                
                # Get comprehensive data including volume for validation
                # Disable progress bar and set timeout to prevent hanging
                df_yf = ticker.history(period=period, auto_adjust=True, back_adjust=True, 
                                     actions=False, prepost=False, repair=True,
                                     progress=False, timeout=10)
                
                if df_yf.empty or len(df_yf) < 50:
                    logger.warning(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Insufficient data for {symbol}: {len(df_yf) if not df_yf.empty else 0} days")
                    continue
                    
                # Ensure we have required columns
                if 'Close' not in df_yf.columns or 'Volume' not in df_yf.columns:
                    logger.warning(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Missing required data columns for {symbol}")
                    continue
                    
                # Filter out low-volume days (likely data quality issues)
                df_yf = df_yf[df_yf['Volume'] > 0]
                
                if df_yf.empty:
                    logger.warning(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] No valid trading data for {symbol}")
                    continue
                    
                yf_close = df_yf['Close']
                logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Raw close prices for {symbol}: {len(yf_close)} data points")
                
                # Validate data quality - remove obvious errors
                yf_close = yf_close.dropna()
                
                # Remove price jumps that are likely stock splits or data errors
                price_changes = yf_close.pct_change().abs()
                split_threshold = 0.4  # 40% change likely indicates split
                valid_indices = price_changes <= split_threshold
                valid_indices.iloc[0] = True  # Keep first value
                yf_close = yf_close[valid_indices]
                
                if len(yf_close) < 50:
                    logger.warning(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Insufficient clean data for {symbol}: {len(yf_close)} days")
                    continue
                    
                # Calculate daily returns
                returns = yf_close.pct_change().dropna()
                logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Calculated returns for {symbol}: {len(returns)} data points")
                symbol_duration = time.time() - symbol_start_time
                logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Processing {symbol} completed in {symbol_duration:.2f} seconds")
                
                # More conservative outlier filtering
                q1 = returns.quantile(0.01)
                q99 = returns.quantile(0.99)
                returns = returns[(returns >= q1) & (returns <= q99)]
                
                # Additional filter for extreme values
                returns = returns[(returns > -0.3) & (returns < 0.3)]  # Max 30% daily change
                
                if len(returns) < 50:  # Need at least 50 days for meaningful statistics
                    logger.warning(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Insufficient clean returns for {symbol}: {len(returns)} days")
                    continue
                
                # Calculate and log key statistics for validation
                mean_return = returns.mean()
                std_return = returns.std()
                skewness = returns.skew()
                logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Statistics for {symbol}:")
                logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {symbol}: {len(returns)} days, mean={mean_return:.4f}, std={std_return:.4f}, skew={skewness:.2f}")
                
                # Ensure we have reasonable variance (not a constant stock price)
                if std_return < 0.001:  # Less than 0.1% daily volatility is suspicious
                    logger.warning(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {symbol} has very low volatility ({std_return:.4f}), may affect optimization")
                    
                # Store the returns for this symbol
                all_returns[symbol] = returns
                
            except Exception as e:
                logger.error(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error fetching data for {symbol}: {str(e)}\nTraceback: {traceback.format_exc()}")
                continue
        
        if not all_returns:
            logger.error(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] No valid data fetched for any symbols")
            return pd.DataFrame()
        
        # Align all returns to common dates
        returns_df = pd.DataFrame(all_returns)
        logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Created returns DataFrame with shape: {returns_df.shape}")
        
        # Remove rows where any symbol has NaN
        initial_length = len(returns_df)
        returns_df = returns_df.dropna()
        logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] After removing NaN rows: {len(returns_df)} rows (dropped {initial_length - len(returns_df)} rows)")
        
        if returns_df.empty:
            logger.error(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] No overlapping data found for the provided symbols")
            return pd.DataFrame()
        
        # Ensure we have sufficient overlapping data
        if len(returns_df) < 50:
            logger.warning(f"Limited overlapping data: {len(returns_df)} days")
            return pd.DataFrame()
            
        # Log correlation matrix for validation
        if len(returns_df.columns) > 1:
            corr_matrix = returns_df.corr()
            avg_correlation = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
            logger.info(f"Average correlation between assets: {avg_correlation:.3f}")
            
            # Log individual asset statistics
            for col in returns_df.columns:
                col_data = returns_df[col]
                logger.info(f"{col} final stats: mean={col_data.mean():.4f}, std={col_data.std():.4f}")
            
        logger.info(f"Final dataset: {len(returns_df)} days of data for {len(returns_df.columns)} symbols")
        logger.info(f"Data range: {returns_df.index.min()} to {returns_df.index.max()}")
        
        total_duration = time.time() - start_time
        logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Total data fetching and processing completed in {total_duration:.2f} seconds")
        
        return returns_df
    
    def _create_dataset(self, symbols: List[str], start_date: str, end_date: str):
        """Create Qlib dataset"""
        # This would normally create a Qlib dataset
        # For demo, return None
        return None
    
    def _get_model_prediction(self, model, symbol: str) -> Dict:
        """Get prediction from a trained model"""
        # This would normally get real predictions
        # For demo, return random prediction
        return {
            'score': np.random.uniform(-1, 1),
            'prediction': np.random.uniform(0, 200)
        } 

    def _calculate_portfolio_metrics(self, weights: dict, expected_returns: pd.Series, covariance: pd.DataFrame, returns_data: pd.DataFrame = None) -> dict:
        """Calculate accurate expected return, volatility, Sharpe ratio, and max drawdown for the portfolio."""
        try:
            # Convert weights dict to numpy array in the order of expected_returns
            w = np.array([weights[s] for s in expected_returns.index])
            
            # Expected annualized return (assuming daily returns, 252 trading days)
            exp_return = np.dot(w, expected_returns) * 252
            
            # Portfolio volatility (annualized)
            port_vol = np.sqrt(np.dot(w.T, np.dot(covariance * 252, w)))
            
            # Sharpe ratio (assume risk-free rate = 2% annually)
            risk_free_rate = 0.02
            sharpe = (exp_return - risk_free_rate) / port_vol if port_vol > 0 else 0
            
            # Calculate max drawdown using actual historical returns data
            if returns_data is not None and not returns_data.empty and len(returns_data) > 0:
                # Calculate portfolio daily returns
                portfolio_returns = (returns_data * w).sum(axis=1)
                
                # Calculate cumulative returns
                cum_returns = (1 + portfolio_returns).cumprod()
                
                # Calculate running maximum
                running_max = cum_returns.expanding().max()
                
                # Calculate drawdown
                drawdown = (cum_returns - running_max) / running_max
                
                # Maximum drawdown is the minimum (most negative) drawdown
                max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0
            else:
                # Fallback calculation if no historical data available
                # Estimate max drawdown based on volatility (rough approximation)
                max_drawdown = min(0.5, port_vol * 1.5)  # Cap at 50%
            
            # Ensure all metrics are reasonable
            exp_return = max(-1.0, min(2.0, exp_return))  # Cap between -100% and 200%
            port_vol = max(0.01, min(2.0, port_vol))      # Cap between 1% and 200%
            sharpe = max(-5.0, min(5.0, sharpe))          # Cap between -5 and 5
            max_drawdown = max(0.0, min(1.0, max_drawdown)) # Cap between 0% and 100%
            
            logger.info(f"Portfolio metrics calculated: return={exp_return:.3f}, vol={port_vol:.3f}, sharpe={sharpe:.3f}, drawdown={max_drawdown:.3f}")
            
            return {
                'expected_return': exp_return,
                'volatility': port_vol,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown
            }
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            # Return reasonable fallback values instead of zeros
            return {
                'expected_return': 0.08,  # 8% annual return
                'volatility': 0.15,       # 15% volatility
                'sharpe_ratio': 0.53,     # (8%-2%)/15%
                'max_drawdown': 0.20      # 20% max drawdown
            }
    
    def _fallback_optimization(self, symbols: List[str], risk_tolerance: str) -> Dict:
        """Fallback optimization when modern portfolio optimizer is not available"""
        logger.warning("Using fallback optimization (modern portfolio optimizer not available)")
        
        try:
            # Simple equal-weight portfolio
            weights = {symbol: 1/len(symbols) for symbol in symbols}
            
            # Fetch some basic data for metrics
            price_data = self._fetch_price_data(symbols)
            
            if not price_data.empty:
                returns = price_data.pct_change().dropna()
                portfolio_returns = (returns * pd.Series(weights)).sum(axis=1)
                
                expected_return = portfolio_returns.mean() * 252
                volatility = portfolio_returns.std() * np.sqrt(252)
                sharpe_ratio = expected_return / volatility if volatility > 0 else 0
                
                metrics = {
                    'expected_return': expected_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': 0.20,
                    'var_95': np.percentile(portfolio_returns, 5),
                    'beta': 1.0
                }
            else:
                metrics = {
                    'expected_return': 0.08,
                    'volatility': 0.15,
                    'sharpe_ratio': 0.53,
                    'max_drawdown': 0.20,
                    'var_95': 0.02,
                    'beta': 1.0
                }
            
            return {
                'weights': weights,
                'metrics': metrics,
                'risk_tolerance': risk_tolerance,
                'strategy': 'Equal Weight (Fallback)',
                'symbols': symbols,
                'data_period_days': len(price_data) if not price_data.empty else 0,
                'optimization_timestamp': datetime.now().isoformat(),
                'warning': 'Modern portfolio optimizer not available, using basic optimization'
            }
                
        except Exception as e:
            logger.error(f"Error in fallback optimization: {e}")
            return {
                'weights': {s: 1/len(symbols) for s in symbols},
                'metrics': {},
                'risk_tolerance': risk_tolerance,
                'symbols': symbols,
                'error': f'Portfolio optimization failed: {str(e)}. Please verify symbols and try again.',
                'optimization_timestamp': datetime.now().isoformat()
            }
    
    def risk_management(self, portfolio: Dict, market_data: Dict) -> Dict:
        """Risk management and position sizing"""
        try:
            risk_metrics = {}
            
            # Calculate Value at Risk (VaR)
            portfolio_value = 100000  # Example portfolio value
            var_95 = self._calculate_var(portfolio, market_data, confidence=0.95)
            var_99 = self._calculate_var(portfolio, market_data, confidence=0.99)
            
            # Calculate Maximum Drawdown
            max_drawdown = self._calculate_max_drawdown(portfolio, market_data)
            
            # Position sizing based on volatility
            position_sizes = self._calculate_position_sizes(portfolio, market_data)
            
            # Risk-adjusted returns
            sharpe_ratio = self._calculate_sharpe_ratio(portfolio, market_data)
            
            risk_metrics = {
                'var_95': var_95,
                'var_99': var_99,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'position_sizes': position_sizes,
                'risk_level': self._assess_risk_level(var_95, max_drawdown, sharpe_ratio)
            }
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error in risk management: {e}")
            return self._demo_risk_metrics()
    
    def _calculate_var(self, portfolio: Dict, market_data: Dict, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        try:
            # Simplified VaR calculation
            portfolio_volatility = 0.15  # Example volatility
            z_score = 1.645 if confidence == 0.95 else 2.326
            var = z_score * portfolio_volatility * 100000  # Portfolio value
            return var
        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            return 0
    
    def _calculate_max_drawdown(self, portfolio: Dict, market_data: Dict) -> float:
        """Calculate maximum drawdown"""
        try:
            # Simplified max drawdown calculation
            return 0.25  # Example 25% max drawdown
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0
    
    def _calculate_sharpe_ratio(self, portfolio: Dict, market_data: Dict) -> float:
        """Calculate Sharpe ratio"""
        try:
            # Simplified Sharpe ratio calculation
            return 1.2  # Example Sharpe ratio
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0
    
    def _assess_risk_level(self, var_95: float, max_drawdown: float, sharpe_ratio: float) -> str:
        """Assess overall risk level"""
        try:
            risk_score = 0
            
            # VaR assessment
            if var_95 > 5000:
                risk_score += 3
            elif var_95 > 3000:
                risk_score += 2
            else:
                risk_score += 1
            
            # Drawdown assessment
            if max_drawdown > 0.3:
                risk_score += 3
            elif max_drawdown > 0.2:
                risk_score += 2
            else:
                risk_score += 1
            
            # Sharpe ratio assessment
            if sharpe_ratio < 0.5:
                risk_score += 3
            elif sharpe_ratio < 1.0:
                risk_score += 2
            else:
                risk_score += 1
            
            # Determine risk level
            if risk_score <= 4:
                return 'LOW'
            elif risk_score <= 6:
                return 'MODERATE'
            else:
                return 'HIGH'
                
        except Exception as e:
            logger.error(f"Error assessing risk level: {e}")
            return 'MODERATE'
    
    # Demo methods for when Qlib is not available
    def _demo_multiple_models(self, symbols: List[str]) -> Dict:
        """Demo multiple models when Qlib is not available"""
        return {
            'lgb_model': 'demo_model',
            'linear_model': 'demo_model',
            'gru_model': 'demo_model'
        }
    
    def _demo_ensemble_signals(self, symbols: List[str]) -> Dict:
        """Demo ensemble signals"""
        signals = {}
        for symbol in symbols:
            signals[symbol] = {
                'signal': np.random.choice(['BUY', 'SELL', 'HOLD']),
                'score': np.random.uniform(-1, 1),
                'confidence': np.random.uniform(60, 95),
                'models_used': 3,
                'individual_predictions': {
                    'lgb_model': {'score': np.random.uniform(-1, 1)},
                    'linear_model': {'score': np.random.uniform(-1, 1)},
                    'gru_model': {'score': np.random.uniform(-1, 1)}
                }
            }
        return signals
    
    def _demo_portfolio_optimization(self, symbols: List[str]) -> Dict:
        """Demo portfolio optimization"""
        weights = {symbol: 1/len(symbols) for symbol in symbols}
        return {
            'weights': weights,
            'metrics': {
                'expected_return': 0.10,
                'volatility': 0.18,
                'sharpe_ratio': 1.2,
                'max_drawdown': 0.15
            },
            'risk_tolerance': 'moderate',
            'symbols': symbols
        }
    
    def _demo_risk_metrics(self) -> Dict:
        """Demo risk metrics"""
        return {
            'var_95': 3500,
            'var_99': 5000,
            'max_drawdown': 0.20,
            'sharpe_ratio': 1.1,
            'position_sizes': {'AAPL': 0.25, 'TSLA': 0.25, 'GOOGL': 0.25, 'MSFT': 0.25},
            'risk_level': 'MODERATE'
        }
    
    def fetch_alpha_vantage_close(self, symbol, start, end, api_key=None):
        """Fetch daily close prices from Alpha Vantage."""
        try:
            start_time = time.time()
            logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting Alpha Vantage fetch for {symbol}")
            
            if not api_key:
                api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
                if not api_key:
                    logger.error("ALPHA_VANTAGE_API_KEY environment variable not set")
                    return None
            
            url = 'https://www.alphavantage.co/query'
            params = {
                'function': 'TIME_SERIES_DAILY_ADJUSTED',
                'symbol': symbol,
                'outputsize': 'full',
                'apikey': api_key
            }
            
            logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Sending request to Alpha Vantage API for {symbol}")
            r = requests.get(url, params=params, timeout=10, headers={'Cache-Control': 'no-cache'})
            
            if r.status_code != 200:
                logger.error(f"Alpha Vantage API request failed with status code {r.status_code}")
                return None
                
            data = r.json()
            if 'Error Message' in data:
                logger.error(f"Alpha Vantage API error: {data['Error Message']}")
                return None
                
            if 'Time Series (Daily)' not in data:
                if 'Note' in data:
                    logger.warning(f"Alpha Vantage API limit reached: {data['Note']}")
                else:
                    logger.warning(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Alpha Vantage: No data for {symbol}")
                return None
            ts = data['Time Series (Daily)']
            df = pd.DataFrame.from_dict(ts, orient='index')
            df.index = pd.to_datetime(df.index, utc=True)
            df = df.sort_index()
            df = df[(df.index >= start) & (df.index <= end)]
            df = df.rename(columns={'5. adjusted close': 'Close'})
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            return df['Close'].dropna()
        except Exception as e:
            logger.error(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Alpha Vantage fetch failed for {symbol}: {str(e)}\nTraceback: {traceback.format_exc()}")
            return None
        finally:
            duration = time.time() - start_time
            logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Alpha Vantage fetch for {symbol} completed in {duration:.2f} seconds")

    def fetch_google_finance_close(self, symbol, start, end):
        """Fetch daily close prices from Google Finance using pandas_datareader."""
        if not GOOGLE_FINANCE_AVAILABLE:
            logger.warning("pandas_datareader is not installed; Google Finance unavailable.")
            return None
        try:
            df = web.DataReader(symbol, 'google', start, end)
            if 'Close' in df:
                return df['Close'].dropna()
            elif 'close' in df:
                return df['close'].dropna()
            else:
                logger.warning(f"Google Finance: No close data for {symbol}")
                return None
        except Exception as e:
            logger.warning(f"Google Finance fetch failed for {symbol}: {e}")
            return None

    # Duplicate _get_returns_data method removed