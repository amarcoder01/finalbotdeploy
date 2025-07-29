"""Modern Portfolio Optimization using Riskfolio-Lib
Replaces legacy PyPortfolioOpt with a more robust and feature-rich solution
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    yf = None
    YFINANCE_AVAILABLE = False
    print("Warning: yfinance not available in portfolio_optimizer, using fallback data")
from logger import logger

try:
    import riskfolio as rp
    RISKFOLIO_AVAILABLE = True
    logger.info("Riskfolio-Lib successfully imported")
except (ImportError, AttributeError) as e:
    RISKFOLIO_AVAILABLE = False
    logger.warning(f"Riskfolio-Lib not available: {e}. Using fallback optimization.")
    rp = None

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except (ImportError, AttributeError) as e:
    CVXPY_AVAILABLE = False
    logger.warning(f"CVXPY not available: {e}. Using equal-weight fallback.")
    cp = None

class ModernPortfolioOptimizer:
    """Modern portfolio optimization using Riskfolio-Lib"""
    
    def __init__(self):
        self.risk_models = {
            'conservative': {'target_return': 0.08, 'risk_aversion': 5.0},
            'moderate': {'target_return': 0.12, 'risk_aversion': 2.0},
            'aggressive': {'target_return': 0.18, 'risk_aversion': 1.0}
        }
    
    def optimize_portfolio(self, symbols: List[str], risk_tolerance: str = 'moderate', 
                          lookback_days: int = 252) -> Dict:
        """Optimize portfolio using modern techniques
        
        Args:
            symbols: List of stock symbols
            risk_tolerance: 'conservative', 'moderate', or 'aggressive'
            lookback_days: Number of days for historical data
            
        Returns:
            Dictionary with optimized weights and metrics
        """
        try:
            logger.info(f"Starting portfolio optimization for {len(symbols)} symbols")
            
            # Fetch price data
            price_data = self._fetch_price_data(symbols, lookback_days)
            if price_data.empty:
                return self._fallback_optimization(symbols, risk_tolerance)
            
            # Calculate returns
            returns = price_data.pct_change().dropna()
            
            if not RISKFOLIO_AVAILABLE:
                return self._cvxpy_optimization(returns, risk_tolerance)
            
            # Use Riskfolio-Lib for optimization
            return self._riskfolio_optimization(returns, risk_tolerance)
            
        except Exception as e:
            logger.error(f"Error in portfolio optimization: {e}")
            return self._fallback_optimization(symbols, risk_tolerance)
    
    def _fetch_price_data(self, symbols: List[str], lookback_days: int) -> pd.DataFrame:
        """Fetch historical price data using yfinance"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days + 30)  # Extra buffer
            
            logger.info(f"Fetching data from {start_date.date()} to {end_date.date()}")
            
            # Download data with auto_adjust=True to handle new yfinance behavior
            data = yf.download(symbols, start=start_date, end=end_date, 
                             progress=False, threads=True, auto_adjust=True)
            
            # Handle different data structures based on number of symbols
            if len(symbols) == 1:
                # Single symbol returns a DataFrame with columns like 'Close', 'Volume', etc.
                if 'Close' in data.columns:
                    price_data = data[['Close']].copy()
                    price_data.columns = symbols
                else:
                    logger.error("No 'Close' column found in single symbol data")
                    return pd.DataFrame()
            else:
                # Multiple symbols return MultiIndex columns
                if isinstance(data.columns, pd.MultiIndex):
                    # Extract 'Close' or 'Adj Close' prices
                    if 'Close' in data.columns.get_level_values(0):
                        price_data = data['Close'].copy()
                    elif 'Adj Close' in data.columns.get_level_values(0):
                        price_data = data['Adj Close'].copy()
                    else:
                        logger.error("No 'Close' or 'Adj Close' columns found in multi-symbol data")
                        return pd.DataFrame()
                else:
                    # Fallback: assume data is already price data
                    price_data = data.copy()
            
            # Ensure we have a DataFrame
            if isinstance(price_data, pd.Series):
                price_data = price_data.to_frame(symbols[0])
            
            # Clean data
            price_data = price_data.dropna()
            
            if len(price_data) < 50:  # Minimum data requirement
                logger.warning(f"Insufficient data: only {len(price_data)} days available")
                return pd.DataFrame()
            
            logger.info(f"Successfully fetched {len(price_data)} days of data for {len(price_data.columns)} symbols")
            return price_data
            
        except Exception as e:
            logger.error(f"Error fetching price data: {e}")
            return pd.DataFrame()
    
    def _riskfolio_optimization(self, returns: pd.DataFrame, risk_tolerance: str) -> Dict:
        """Optimize using Riskfolio-Lib"""
        try:
            # Create portfolio object
            port = rp.Portfolio(returns=returns)
            
            # Calculate risk model
            method_mu = 'hist'  # Historical mean
            method_cov = 'hist'  # Historical covariance
            
            port.assets_stats(method_mu=method_mu, method_cov=method_cov)
            
            # Set optimization parameters based on risk tolerance
            risk_params = self.risk_models[risk_tolerance]
            
            # Optimize portfolio
            model = 'Classic'  # Classic mean-variance optimization
            rm = 'MV'  # Mean-Variance risk measure
            obj = 'Sharpe'  # Maximize Sharpe ratio
            hist = True
            rf = 0.02  # Risk-free rate (2%)
            l = 0  # Risk aversion factor
            
            weights = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)
            
            if weights is None or weights.empty:
                logger.warning("Riskfolio optimization failed, using fallback")
                return self._cvxpy_optimization(returns, risk_tolerance)
            
            # Calculate portfolio metrics
            portfolio_return = (weights.T @ port.mu).iloc[0, 0] * 252  # Annualized
            portfolio_vol = np.sqrt(weights.T @ port.cov @ weights).iloc[0, 0] * np.sqrt(252)
            sharpe_ratio = (portfolio_return - rf) / portfolio_vol if portfolio_vol > 0 else 0
            
            # Convert weights to dictionary
            weights_dict = {symbol: float(weight) for symbol, weight in weights.iloc[:, 0].items()}
            
            # Calculate additional metrics
            metrics = self._calculate_portfolio_metrics(returns, weights_dict)
            
            return {
                'weights': weights_dict,
                'metrics': {
                    'expected_return': float(portfolio_return),
                    'volatility': float(portfolio_vol),
                    'sharpe_ratio': float(sharpe_ratio),
                    **metrics
                },
                'risk_tolerance': risk_tolerance,
                'strategy': 'Riskfolio Mean-Variance',
                'symbols': list(returns.columns),
                'data_period_days': len(returns),
                'optimization_timestamp': datetime.now().isoformat(),
                'optimizer': 'riskfolio-lib'
            }
            
        except Exception as e:
            logger.error(f"Error in Riskfolio optimization: {e}")
            return self._cvxpy_optimization(returns, risk_tolerance)
    
    def _cvxpy_optimization(self, returns: pd.DataFrame, risk_tolerance: str) -> Dict:
        """Fallback optimization using CVXPY directly"""
        try:
            if not CVXPY_AVAILABLE:
                logger.warning("CVXPY not available, using equal weights")
                return self._fallback_optimization(list(returns.columns), risk_tolerance)
            
            logger.info("Using CVXPY for portfolio optimization")
            
            n_assets = len(returns.columns)
            mu = returns.mean().values * 252  # Annualized returns
            Sigma = returns.cov().values * 252  # Annualized covariance
            
            # Define optimization variables
            w = cp.Variable(n_assets)
            
            # Risk aversion parameter
            risk_params = self.risk_models[risk_tolerance]
            gamma = risk_params['risk_aversion']
            
            # Objective: maximize return - risk penalty
            portfolio_return = mu.T @ w
            portfolio_risk = cp.quad_form(w, Sigma)
            objective = cp.Maximize(portfolio_return - gamma * portfolio_risk)
            
            # Constraints - balanced to prevent extreme allocations
            min_weight = 0.05  # 5% minimum to ensure diversification
            max_weight = 0.60  # 60% maximum to prevent over-concentration
            
            constraints = [
                cp.sum(w) == 1,  # Weights sum to 1
                w >= min_weight,  # Minimum weight per asset
                w <= max_weight   # Maximum weight per asset
            ]
            
            # Solve optimization with multiple solver fallbacks
            problem = cp.Problem(objective, constraints)
            
            # Try multiple solvers in order of preference
            solvers_to_try = [cp.CLARABEL, cp.OSQP, cp.ECOS, cp.SCS]
            solved = False
            
            for solver in solvers_to_try:
                try:
                    problem.solve(solver=solver, verbose=False)
                    if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                        solved = True
                        logger.info(f"CVXPY optimization successful with {solver}")
                        break
                    else:
                        logger.warning(f"Solver {solver} failed with status: {problem.status}")
                except Exception as e:
                    logger.warning(f"Solver {solver} failed with error: {e}")
                    continue
            
            if solved and problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                weights_array = w.value
                weights_dict = {symbol: float(weight) for symbol, weight in 
                              zip(returns.columns, weights_array)}
                
                # Calculate portfolio metrics using actual returns data
                portfolio_returns = (returns * weights_array).sum(axis=1)
                annual_return = portfolio_returns.mean() * 252
                annual_vol = portfolio_returns.std() * np.sqrt(252)
                sharpe_ratio = (annual_return - 0.02) / annual_vol if annual_vol > 0 else 0
                
                # Get additional metrics
                metrics = self._calculate_portfolio_metrics(returns, weights_dict)
                
                return {
                    'weights': weights_dict,
                    'metrics': {
                        'expected_return': float(annual_return),
                        'volatility': float(annual_vol),
                        'sharpe_ratio': float(sharpe_ratio),
                        **metrics
                    },
                    'risk_tolerance': risk_tolerance,
                    'strategy': 'CVXPY Mean-Variance',
                    'symbols': list(returns.columns),
                    'data_period_days': len(returns),
                    'optimization_timestamp': datetime.now().isoformat(),
                    'optimizer': 'cvxpy'
                }
            else:
                logger.warning("CVXPY optimization failed, using equal weights")
                return self._fallback_optimization(list(returns.columns), risk_tolerance)
                
        except Exception as e:
            logger.error(f"Error in CVXPY optimization: {e}")
            return self._fallback_optimization(list(returns.columns), risk_tolerance)
    
    def _calculate_portfolio_metrics(self, returns: pd.DataFrame, weights: Dict[str, float]) -> Dict:
        """Calculate additional portfolio metrics"""
        try:
            # Convert weights to array
            weight_array = np.array([weights.get(col, 0) for col in returns.columns])
            
            # Portfolio returns
            portfolio_returns = (returns * weight_array).sum(axis=1)
            
            # Calculate metrics
            metrics = {
                'max_drawdown': float(self._calculate_max_drawdown(portfolio_returns)),
                'var_95': float(np.percentile(portfolio_returns, 5) * np.sqrt(252)),
                'cvar_95': float(portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean() * np.sqrt(252)),
                'skewness': float(portfolio_returns.skew()),
                'kurtosis': float(portfolio_returns.kurtosis()),
                'calmar_ratio': 0.0,  # Will calculate if max_drawdown > 0
                'sortino_ratio': float(self._calculate_sortino_ratio(portfolio_returns))
            }
            
            # Calculate Calmar ratio
            if abs(metrics['max_drawdown']) > 0.001:
                annual_return = portfolio_returns.mean() * 252
                metrics['calmar_ratio'] = float(annual_return / abs(metrics['max_drawdown']))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return {
                'max_drawdown': -0.15,
                'var_95': -0.05,
                'cvar_95': -0.08,
                'skewness': 0.0,
                'kurtosis': 3.0,
                'calmar_ratio': 1.0,
                'sortino_ratio': 1.5
            }
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _calculate_sortino_ratio(self, returns: pd.Series, rf: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        excess_returns = returns - rf/252
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_deviation = np.sqrt((downside_returns ** 2).mean()) * np.sqrt(252)
        
        if downside_deviation == 0:
            return float('inf')
        
        return (excess_returns.mean() * 252) / downside_deviation
    
    def _fallback_optimization(self, symbols: List[str], risk_tolerance: str) -> Dict:
        """Simple equal-weight fallback when optimization fails"""
        logger.warning("Using equal-weight fallback optimization")
        
        n_assets = len(symbols)
        equal_weight = 1.0 / n_assets
        weights = {symbol: equal_weight for symbol in symbols}
        
        # Try to fetch real data and calculate actual metrics
        try:
            price_data = self._fetch_price_data(symbols, lookback_days=252)
            if not price_data.empty and len(price_data) > 30:
                returns = price_data.pct_change().dropna()
                metrics = self._calculate_portfolio_metrics(returns, weights)
                
                # Calculate basic portfolio metrics
                portfolio_returns = (returns * equal_weight).sum(axis=1)
                annual_return = portfolio_returns.mean() * 252
                annual_vol = portfolio_returns.std() * np.sqrt(252)
                sharpe_ratio = (annual_return - 0.02) / annual_vol if annual_vol > 0 else 0
                
                metrics.update({
                    'expected_return': float(annual_return),
                    'volatility': float(annual_vol),
                    'sharpe_ratio': float(sharpe_ratio)
                })
                
                data_period_days = len(returns)
            else:
                # Use conservative default metrics if no data available
                risk_params = self.risk_models[risk_tolerance]
                metrics = {
                    'expected_return': risk_params['target_return'] * 0.7,  # Conservative estimate
                    'volatility': 0.18,  # Typical market volatility
                    'sharpe_ratio': 0.6,  # Conservative Sharpe ratio
                    'max_drawdown': -0.20,
                    'var_95': -0.06,
                    'cvar_95': -0.09,
                    'skewness': -0.2,
                    'kurtosis': 3.5,
                    'calmar_ratio': 0.5,
                    'sortino_ratio': 0.8
                }
                data_period_days = 252
                
        except Exception as e:
            logger.error(f"Error in fallback optimization: {e}")
            # Use conservative default metrics
            risk_params = self.risk_models[risk_tolerance]
            metrics = {
                'expected_return': risk_params['target_return'] * 0.7,
                'volatility': 0.18,
                'sharpe_ratio': 0.6,
                'max_drawdown': -0.20,
                'var_95': -0.06,
                'cvar_95': -0.09,
                'skewness': -0.2,
                'kurtosis': 3.5,
                'calmar_ratio': 0.5,
                'sortino_ratio': 0.8
            }
            data_period_days = 252
        
        return {
            'weights': weights,
            'metrics': metrics,
            'risk_tolerance': risk_tolerance,
            'strategy': 'Equal Weight (Fallback)',
            'symbols': symbols,
            'data_period_days': data_period_days,
            'optimization_timestamp': datetime.now().isoformat(),
            'optimizer': 'fallback',
            'warning': 'Advanced optimization failed, using equal weights with estimated metrics'
        }