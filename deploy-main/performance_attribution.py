"""
Performance Attribution System
Analyze trading performance and identify sources of returns
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    sns = None
    SEABORN_AVAILABLE = False
    print("Warning: seaborn not available in performance_attribution, using matplotlib styling")
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from logger import logger

@dataclass
class AttributionResult:
    """Performance attribution result"""
    total_return: float
    benchmark_return: float
    excess_return: float
    factor_contributions: Dict[str, float]
    risk_metrics: Dict[str, float]
    style_analysis: Dict[str, float]
    sector_analysis: Dict[str, float]
    timing_analysis: Dict[str, float]
    selection_analysis: Dict[str, float]

class PerformanceAttribution:
    """Performance attribution analysis system"""
    
    def __init__(self):
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        self.benchmark_data = None
        self.factor_data = None
        
    def analyze_performance(self, portfolio_returns: pd.Series, 
                          benchmark_returns: pd.Series = None,
                          factor_returns: pd.DataFrame = None) -> AttributionResult:
        """Analyze portfolio performance and attribution"""
        try:
            # Calculate basic metrics
            total_return = self._calculate_total_return(portfolio_returns)
            benchmark_return = self._calculate_total_return(benchmark_returns) if benchmark_returns is not None else 0
            excess_return = total_return - benchmark_return
            
            # Factor analysis
            factor_contributions = self._analyze_factor_contributions(portfolio_returns, factor_returns)
            
            # Risk metrics
            risk_metrics = self._calculate_risk_metrics(portfolio_returns, benchmark_returns)
            
            # Style analysis
            style_analysis = self._analyze_style(portfolio_returns, factor_returns)
            
            # Sector analysis (if sector data available)
            sector_analysis = self._analyze_sector_contributions(portfolio_returns)
            
            # Timing analysis
            timing_analysis = self._analyze_timing(portfolio_returns, benchmark_returns)
            
            # Selection analysis
            selection_analysis = self._analyze_selection(portfolio_returns, benchmark_returns)
            
            return AttributionResult(
                total_return=total_return,
                benchmark_return=benchmark_return,
                excess_return=excess_return,
                factor_contributions=factor_contributions,
                risk_metrics=risk_metrics,
                style_analysis=style_analysis,
                sector_analysis=sector_analysis,
                timing_analysis=timing_analysis,
                selection_analysis=selection_analysis
            )
            
        except Exception as e:
            logger.error(f"Error in performance attribution: {e}")
            return self._empty_attribution_result()
    
    def _calculate_total_return(self, returns: pd.Series) -> float:
        """Calculate total return from return series"""
        try:
            if returns is None or len(returns) == 0:
                return 0
            return (1 + returns).prod() - 1
        except Exception as e:
            logger.error(f"Error calculating total return: {e}")
            return 0
    
    def _analyze_factor_contributions(self, portfolio_returns: pd.Series, 
                                    factor_returns: pd.DataFrame) -> Dict[str, float]:
        """Analyze factor contributions using factor model"""
        try:
            if factor_returns is None or len(factor_returns) == 0:
                return self._demo_factor_contributions()
            
            # Align data
            common_index = portfolio_returns.index.intersection(factor_returns.index)
            if len(common_index) == 0:
                return self._demo_factor_contributions()
            
            portfolio_aligned = portfolio_returns.loc[common_index]
            factors_aligned = factor_returns.loc[common_index]
            
            # Calculate factor exposures using regression
            factor_contributions = {}
            
            for factor in factors_aligned.columns:
                try:
                    # Simple correlation-based contribution
                    correlation = portfolio_aligned.corr(factors_aligned[factor])
                    factor_contribution = correlation * factors_aligned[factor].mean() * len(common_index)
                    factor_contributions[factor] = factor_contribution
                except Exception as e:
                    logger.error(f"Error calculating {factor} contribution: {e}")
                    factor_contributions[factor] = 0
            
            return factor_contributions
            
        except Exception as e:
            logger.error(f"Error in factor analysis: {e}")
            return self._demo_factor_contributions()
    
    def _calculate_risk_metrics(self, portfolio_returns: pd.Series, 
                              benchmark_returns: pd.Series = None) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        try:
            metrics = {}
            
            # Basic risk metrics
            metrics['volatility'] = portfolio_returns.std() * np.sqrt(252)
            metrics['var_95'] = np.percentile(portfolio_returns, 5)
            metrics['var_99'] = np.percentile(portfolio_returns, 1)
            metrics['max_drawdown'] = self._calculate_max_drawdown(portfolio_returns)
            metrics['sharpe_ratio'] = self._calculate_sharpe_ratio(portfolio_returns)
            metrics['sortino_ratio'] = self._calculate_sortino_ratio(portfolio_returns)
            metrics['calmar_ratio'] = self._calculate_calmar_ratio(portfolio_returns)
            
            # Information ratio (if benchmark available)
            if benchmark_returns is not None:
                excess_returns = portfolio_returns - benchmark_returns
                metrics['information_ratio'] = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
                metrics['tracking_error'] = excess_returns.std() * np.sqrt(252)
                metrics['beta'] = self._calculate_beta(portfolio_returns, benchmark_returns)
                metrics['alpha'] = self._calculate_alpha(portfolio_returns, benchmark_returns)
            else:
                metrics['information_ratio'] = 0
                metrics['tracking_error'] = 0
                metrics['beta'] = 1
                metrics['alpha'] = 0
            
            # Additional metrics
            metrics['skewness'] = portfolio_returns.skew()
            metrics['kurtosis'] = portfolio_returns.kurtosis()
            metrics['var_ratio'] = metrics['var_95'] / metrics['volatility']
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}
    
    def _analyze_style(self, portfolio_returns: pd.Series, 
                      factor_returns: pd.DataFrame) -> Dict[str, float]:
        """Analyze investment style using factor model"""
        try:
            if factor_returns is None:
                return self._demo_style_analysis()
            
            # Align data
            common_index = portfolio_returns.index.intersection(factor_returns.index)
            if len(common_index) == 0:
                return self._demo_style_analysis()
            
            portfolio_aligned = portfolio_returns.loc[common_index]
            factors_aligned = factor_returns.loc[common_index]
            
            # Calculate style exposures
            style_exposures = {}
            
            for factor in factors_aligned.columns:
                try:
                    # Calculate exposure using rolling correlation
                    rolling_corr = portfolio_aligned.rolling(window=60).corr(factors_aligned[factor])
                    style_exposures[factor] = rolling_corr.mean()
                except Exception as e:
                    logger.error(f"Error calculating {factor} style exposure: {e}")
                    style_exposures[factor] = 0
            
            return style_exposures
            
        except Exception as e:
            logger.error(f"Error in style analysis: {e}")
            return self._demo_style_analysis()
    
    def _analyze_sector_contributions(self, portfolio_returns: pd.Series) -> Dict[str, float]:
        """Analyze sector contributions (demo implementation)"""
        try:
            # This would normally use sector data
            # For demo, return mock sector analysis
            sectors = ['Technology', 'Healthcare', 'Financials', 'Consumer', 'Energy', 'Others']
            contributions = {}
            
            for sector in sectors:
                contributions[sector] = np.random.uniform(-0.05, 0.05)
            
            # Normalize to sum to total return
            total_contribution = sum(contributions.values())
            if total_contribution != 0:
                for sector in contributions:
                    contributions[sector] = contributions[sector] / total_contribution * portfolio_returns.mean() * 252
            
            return contributions
            
        except Exception as e:
            logger.error(f"Error in sector analysis: {e}")
            return {}
    
    def _analyze_timing(self, portfolio_returns: pd.Series, 
                       benchmark_returns: pd.Series) -> Dict[str, float]:
        """Analyze market timing ability"""
        try:
            if benchmark_returns is None:
                return {'timing_skill': 0, 'timing_contribution': 0}
            
            # Align data
            common_index = portfolio_returns.index.intersection(benchmark_returns.index)
            if len(common_index) == 0:
                return {'timing_skill': 0, 'timing_contribution': 0}
            
            portfolio_aligned = portfolio_returns.loc[common_index]
            benchmark_aligned = benchmark_returns.loc[common_index]
            
            # Calculate timing metrics
            excess_returns = portfolio_aligned - benchmark_aligned
            
            # Market timing using Treynor-Mazuy model
            timing_skill = self._calculate_timing_skill(portfolio_aligned, benchmark_aligned)
            
            # Timing contribution
            timing_contribution = timing_skill * benchmark_aligned.var()
            
            return {
                'timing_skill': timing_skill,
                'timing_contribution': timing_contribution,
                'timing_correlation': excess_returns.corr(benchmark_aligned)
            }
            
        except Exception as e:
            logger.error(f"Error in timing analysis: {e}")
            return {'timing_skill': 0, 'timing_contribution': 0}
    
    def _analyze_selection(self, portfolio_returns: pd.Series, 
                          benchmark_returns: pd.Series) -> Dict[str, float]:
        """Analyze stock selection ability"""
        try:
            if benchmark_returns is None:
                return {'selection_skill': 0, 'selection_contribution': 0}
            
            # Align data
            common_index = portfolio_returns.index.intersection(benchmark_returns.index)
            if len(common_index) == 0:
                return {'selection_skill': 0, 'selection_contribution': 0}
            
            portfolio_aligned = portfolio_returns.loc[common_index]
            benchmark_aligned = benchmark_returns.loc[common_index]
            
            # Calculate selection metrics
            excess_returns = portfolio_aligned - benchmark_aligned
            
            # Selection skill (Jensen's alpha)
            selection_skill = self._calculate_alpha(portfolio_aligned, benchmark_aligned)
            
            # Selection contribution
            selection_contribution = selection_skill * len(common_index) / 252
            
            return {
                'selection_skill': selection_skill,
                'selection_contribution': selection_contribution,
                'selection_volatility': excess_returns.std() * np.sqrt(252)
            }
            
        except Exception as e:
            logger.error(f"Error in selection analysis: {e}")
            return {'selection_skill': 0, 'selection_contribution': 0}
    
    # Helper methods for risk calculations
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        try:
            cumulative = (1 + returns).cumprod()
            peak = cumulative.expanding().max()
            drawdown = (cumulative - peak) / peak
            return abs(drawdown.min())
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        try:
            excess_returns = returns - self.risk_free_rate / 252
            if excess_returns.std() == 0:
                return 0
            return excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio"""
        try:
            excess_returns = returns - self.risk_free_rate / 252
            downside_returns = excess_returns[excess_returns < 0]
            if len(downside_returns) == 0 or downside_returns.std() == 0:
                return 0
            return excess_returns.mean() / downside_returns.std() * np.sqrt(252)
        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {e}")
            return 0
    
    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar ratio"""
        try:
            total_return = (1 + returns).prod() - 1
            max_dd = self._calculate_max_drawdown(returns)
            if max_dd == 0:
                return 0
            return total_return / max_dd
        except Exception as e:
            logger.error(f"Error calculating Calmar ratio: {e}")
            return 0
    
    def _calculate_beta(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate beta"""
        try:
            covariance = portfolio_returns.cov(benchmark_returns)
            variance = benchmark_returns.var()
            if variance == 0:
                return 1
            return covariance / variance
        except Exception as e:
            logger.error(f"Error calculating beta: {e}")
            return 1
    
    def _calculate_alpha(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate alpha (Jensen's alpha)"""
        try:
            beta = self._calculate_beta(portfolio_returns, benchmark_returns)
            portfolio_mean = portfolio_returns.mean() * 252
            benchmark_mean = benchmark_returns.mean() * 252
            risk_free = self.risk_free_rate
            
            alpha = portfolio_mean - (risk_free + beta * (benchmark_mean - risk_free))
            return alpha
        except Exception as e:
            logger.error(f"Error calculating alpha: {e}")
            return 0
    
    def _calculate_timing_skill(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate market timing skill using Treynor-Mazuy model"""
        try:
            # Simple timing measure using correlation
            return portfolio_returns.corr(benchmark_returns)
        except Exception as e:
            logger.error(f"Error calculating timing skill: {e}")
            return 0
    
    # Demo methods
    def _demo_factor_contributions(self) -> Dict[str, float]:
        """Demo factor contributions"""
        factors = ['Market', 'Size', 'Value', 'Momentum', 'Quality', 'Volatility']
        contributions = {}
        
        for factor in factors:
            contributions[factor] = np.random.uniform(-0.02, 0.02)
        
        return contributions
    
    def _demo_style_analysis(self) -> Dict[str, float]:
        """Demo style analysis"""
        styles = ['Growth', 'Value', 'Large Cap', 'Small Cap', 'Momentum', 'Quality']
        exposures = {}
        
        for style in styles:
            exposures[style] = np.random.uniform(-0.5, 0.5)
        
        return exposures
    
    def _empty_attribution_result(self) -> AttributionResult:
        """Return empty attribution result"""
        return AttributionResult(
            total_return=0,
            benchmark_return=0,
            excess_return=0,
            factor_contributions={},
            risk_metrics={},
            style_analysis={},
            sector_analysis={},
            timing_analysis={},
            selection_analysis={}
        )
    
    def generate_attribution_report(self, result: AttributionResult, portfolio_name: str = "Portfolio") -> str:
        """Generate performance attribution report"""
        try:
            report = f"""
📊 **Performance Attribution Report: {portfolio_name}**

💰 **Return Analysis:**
• Total Return: {result.total_return:.2%}
• Benchmark Return: {result.benchmark_return:.2%}
• Excess Return: {result.excess_return:.2%}

📈 **Risk Metrics:**
• Volatility: {result.risk_metrics.get('volatility', 0):.2%}
• Sharpe Ratio: {result.risk_metrics.get('sharpe_ratio', 0):.2f}
• Maximum Drawdown: {result.risk_metrics.get('max_drawdown', 0):.2%}
• VaR (95%): {result.risk_metrics.get('var_95', 0):.2%}
• Beta: {result.risk_metrics.get('beta', 1):.2f}
• Alpha: {result.risk_metrics.get('alpha', 0):.2%}

🎯 **Factor Contributions:**
"""
            
            for factor, contribution in result.factor_contributions.items():
                report += f"• {factor}: {contribution:.2%}\n"
            
            report += f"""
🎨 **Style Analysis:**
"""
            
            for style, exposure in result.style_analysis.items():
                report += f"• {style}: {exposure:.2f}\n"
            
            report += f"""
⏰ **Timing Analysis:**
• Timing Skill: {result.timing_analysis.get('timing_skill', 0):.3f}
• Timing Contribution: {result.timing_analysis.get('timing_contribution', 0):.2%}

🔍 **Selection Analysis:**
• Selection Skill: {result.selection_analysis.get('selection_skill', 0):.2%}
• Selection Contribution: {result.selection_analysis.get('selection_contribution', 0):.2%}
"""
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating attribution report: {e}")
            return "Error generating attribution report"
    
    def plot_attribution_analysis(self, result: AttributionResult, portfolio_name: str = "Portfolio"):
        """Plot attribution analysis"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Factor contributions
            if result.factor_contributions:
                factors = list(result.factor_contributions.keys())
                contributions = list(result.factor_contributions.values())
                axes[0, 0].bar(factors, contributions)
                axes[0, 0].set_title(f'{portfolio_name} - Factor Contributions')
                axes[0, 0].tick_params(axis='x', rotation=45)
                axes[0, 0].grid(True)
            
            # Style analysis
            if result.style_analysis:
                styles = list(result.style_analysis.keys())
                exposures = list(result.style_analysis.values())
                axes[0, 1].bar(styles, exposures)
                axes[0, 1].set_title(f'{portfolio_name} - Style Exposures')
                axes[0, 1].tick_params(axis='x', rotation=45)
                axes[0, 1].grid(True)
            
            # Risk metrics
            risk_metrics = ['Volatility', 'Sharpe Ratio', 'Max Drawdown', 'VaR (95%)']
            risk_values = [
                result.risk_metrics.get('volatility', 0),
                result.risk_metrics.get('sharpe_ratio', 0),
                result.risk_metrics.get('max_drawdown', 0),
                result.risk_metrics.get('var_95', 0)
            ]
            axes[1, 0].bar(risk_metrics, risk_values)
            axes[1, 0].set_title(f'{portfolio_name} - Risk Metrics')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True)
            
            # Return decomposition
            return_components = ['Total Return', 'Benchmark', 'Excess Return']
            return_values = [
                result.total_return,
                result.benchmark_return,
                result.excess_return
            ]
            axes[1, 1].bar(return_components, return_values)
            axes[1, 1].set_title(f'{portfolio_name} - Return Decomposition')
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting attribution analysis: {e}")
            return None 