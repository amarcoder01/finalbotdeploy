import os
import sys
import qlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from qlib.data import D
from qlib.contrib.data.handler import Alpha158
from qlib.config import REG_US
from lightgbm import LGBMRegressor
from pypfopt import EfficientFrontier, risk_models, expected_returns
from datetime import datetime

# ========== QLIB PARALLEL PATCH (for ParallelExt _backend_args bug) ========== #
import importlib
try:
    paral_mod = importlib.import_module('qlib.utils.paral')
    if hasattr(paral_mod, 'ParallelExt'):
        ParallelExt = paral_mod.ParallelExt
        if not hasattr(ParallelExt, '_backend_args'):
            ParallelExt._backend_args = {}
except Exception:
    pass
# ========== END PATCH ========== #

if __name__ == "__main__":
    # ========== CONFIGURATION ========== #
    QLIB_DATA_DIR = "qlib_data/us_data"
    RESULTS_DIR = "results"
    START_DATE = "2015-01-01"
    END_DATE = "2020-11-10"
    UNIVERSE = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "UNH"]  # Top US stocks

    # ========== UTILS ========== #
    def bold(msg):
        return f"\033[1m{msg}\033[0m"

    def print_section(title):
        print(f"\n{'='*60}\n{bold(title)}\n{'='*60}")

    def ensure_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)

    # ========== 1. INIT QLIB ========== #
    print_section("Initializing Qlib and Environment")
    try:
        qlib.init(provider_uri=QLIB_DATA_DIR, region=REG_US)
        ensure_dir(RESULTS_DIR)
        print(bold(f"Qlib initialized with data: {QLIB_DATA_DIR}"))
    except Exception as e:
        print(bold(f"[ERROR] Failed to initialize Qlib: {e}"))
        sys.exit(1)

    # ========== 2. LOAD DATA ========== #
    print_section("Loading Data from Qlib")
    try:
        from qlib.data import D
        # Manually read instruments from the file since D.instruments() has issues
        instruments_file = os.path.join(QLIB_DATA_DIR, "instruments", "all.txt")
        all_instruments = []
        
        if os.path.exists(instruments_file):
            with open(instruments_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Extract symbol from line format: "SYMBOL\tSTART_DATE\tEND_DATE"
                        symbol = line.split('\t')[0]
                        all_instruments.append(symbol)
        else:
            print(bold(f"[ERROR] Instruments file not found: {instruments_file}"))
            sys.exit(1)
        
        print(bold(f"Available instruments in Qlib data (first 20): {all_instruments[:20]} ... (total {len(all_instruments)})"))
        # Use a smaller test universe for faster processing
        test_universe = ["AAPL", "MSFT", "GOOGL"]
        # Filter to only include stocks that exist in our data
        test_universe = [stock for stock in test_universe if stock in all_instruments]
        if not test_universe:
            # Fallback to first 3 available stocks if none of our preferred stocks are available
            test_universe = all_instruments[:3]
        print(bold(f"Using test universe: {test_universe}"))
        handler = None
        df = None
        # Try custom universe dict format
        try:
            print(bold("Attempting to load data with custom universe dict..."))
            custom_universe = {"market": "us", "instruments": list(test_universe)}
            print(bold(f"Custom universe: {custom_universe}"))
            handler = Alpha158(instruments=custom_universe, start_time=START_DATE, end_time=END_DATE)
            print(bold("Handler created, fetching data..."))
            df = handler.fetch(col_set="feature")
            print(bold(f"✓ Successfully loaded data with custom universe dict. Shape: {df.shape}"))
        except Exception as e:
            print(bold(f"[WARN] Custom universe dict failed: {e}. Trying direct list."))
            try:
                print(bold("Attempting to load data with direct list..."))
                print(bold(f"Direct list: {test_universe}"))
                handler = Alpha158(instruments=test_universe, start_time=START_DATE, end_time=END_DATE)
                print(bold("Handler created, fetching data..."))
                df = handler.fetch(col_set="feature")
                print(bold(f"✓ Successfully loaded data with direct list. Shape: {df.shape}"))
            except Exception as e2:
                print(bold(f"[WARN] Direct list failed: {e2}. Trying 'sp500' pool."))
                try:
                    print(bold("Attempting to load data with 'sp500' pool..."))
                    handler = Alpha158(instruments="sp500", start_time=START_DATE, end_time=END_DATE)
                    print(bold("Handler created, fetching data..."))
                    df = handler.fetch(col_set="feature")
                    print(bold(f"✓ Successfully loaded data with 'sp500' pool. Shape: {df.shape}"))
                except Exception as e3:
                    print(bold(f"[ERROR] All data loading attempts failed. Last error: {e3}"))
                    print(bold("Please check your Qlib data version, documentation, and available pools."))
                    sys.exit(1)
        print(bold(f"Loaded data for {len(test_universe)} stocks from {START_DATE} to {END_DATE}"))
        print(bold(f"Shape of loaded DataFrame: {df.shape}"))
        print(bold(f"Available columns in loaded data: {list(df.columns)}"))
        # Print first few rows of all price features
        price_features = [c for c in df.columns if any(p in c.upper() for p in ["VWAP", "OPEN", "HIGH", "LOW", "MA5", "MA10"])]
        print(bold("Sample of price features for test universe:"))
        print(df[price_features].head(10))
        # Check if all price features are NaN
        if all(df[c].isna().all() for c in price_features):
            print(bold("[ERROR] All price features are NaN for the test universe. Please check your Qlib data or try a different date range/universe."))
            sys.exit(1)
    except Exception as e:
        print(bold(f"[ERROR] Failed to load data: {e}"))
        sys.exit(1)

    # ========== 3. FEATURE ENGINEERING ========== #
    print_section("Feature Engineering")
    try:
        # Check available features and their NaN counts
        print(bold("Available features and their NaN counts:"))
        for col in df.columns[:20]:  # Show first 20 features
            nan_count = df[col].isna().sum()
            print(f"  {col}: {nan_count} NaNs out of {len(df)}")
        
        # Select features with minimal NaN values
        feature_nan_counts = {col: df[col].isna().sum() for col in df.columns}
        valid_features = [col for col, nan_count in feature_nan_counts.items() if nan_count < len(df) * 0.5]
        
        # --- Updated Feature Selection for Alpha158 ---
        price_priority = ["VWAP0", "OPEN0", "HIGH0", "LOW0", "MA5", "MA10"]
        volume_priority = ["VMA5", "VMA10", "VSTD5", "VSUMP5"]
        price_cols = [c for c in valid_features if any(p == c.upper() for p in price_priority)]
        volume_cols = [c for c in valid_features if any(v == c.upper() for v in volume_priority)]
        
        # If no priority features found, use any valid price/volume features
        if not price_cols:
            price_cols = [col for col in valid_features if any(x in col.upper() for x in ['CLOSE', 'OPEN', 'HIGH', 'LOW', 'VWAP'])]
        if not volume_cols:
            volume_cols = [col for col in valid_features if any(x in col.upper() for x in ['VOLUME', 'VMA'])]
            
        def pick_best(cols, priority):
            for p in priority:
                for c in cols:
                    if p == c.upper():
                        return c
            return cols[0] if cols else None
        
        price_col = pick_best(price_cols, price_priority)
        volume_col = pick_best(volume_cols, volume_priority)
        
        if price_col is None:
            print(bold(f"[ERROR] No valid price features found. Available features: {list(df.columns)[:10]}..."))
            sys.exit(1)
        if volume_col is None:
            print(bold("[WARN] No valid volume features found, using first available feature as proxy"))
            volume_col = valid_features[0] if valid_features else df.columns[0]
            
        print(bold(f"Selected price feature: {price_col} (NaN count: {feature_nan_counts[price_col]})"))
        print(bold(f"Selected volume feature: {volume_col} (NaN count: {feature_nan_counts[volume_col]})"))
        # --- Core Features with min_periods to handle NaN ---
        df["return"] = df[price_col].groupby("instrument").pct_change(fill_method=None)
        df["volatility"] = df[price_col].groupby("instrument").rolling(5, min_periods=1).std().reset_index(level=0, drop=True)
        df["volume_z"] = df[volume_col].groupby("instrument").transform(lambda x: (x - x.rolling(10, min_periods=1).mean()) / x.rolling(10, min_periods=1).std().fillna(1))
        # --- Additional Engineered Features (smaller windows with min_periods) ---
        df["momentum_5"] = df[price_col].groupby("instrument").pct_change(periods=1, fill_method=None)
        df["roll_min_5"] = df[price_col].groupby("instrument").rolling(5, min_periods=1).min().reset_index(level=0, drop=True)
        df["roll_max_5"] = df[price_col].groupby("instrument").rolling(5, min_periods=1).max().reset_index(level=0, drop=True)
        df["price_vol_ratio"] = df[price_col] / (df[volume_col] + 1e-9)
        df["price_z"] = df[price_col].groupby("instrument").transform(lambda x: (x - x.rolling(10, min_periods=1).mean()) / x.rolling(10, min_periods=1).std().fillna(1))
        # --- Use all Alpha158 features for ML (except leakage/non-numeric) ---
        ignore_cols = set(["return", "volatility", "volume_z", "momentum_5", "roll_min_5", "roll_max_5", "price_vol_ratio", "price_z"])
        ml_features = [c for c in df.columns if c not in ignore_cols and df[c].dtype.kind in 'fi']
        ml_features += ["return", "volatility", "volume_z", "momentum_5", "roll_min_5", "roll_max_5", "price_vol_ratio", "price_z"]
        ml_features = list(dict.fromkeys(ml_features))
        # --- Additional diagnostics before dropna ---
        print(bold(f"NaN count in 'return': {df['return'].isna().sum()}"))
        print(bold(f"NaN count in 'volatility': {df['volatility'].isna().sum()}"))
        print(bold(f"NaN count in 'volume_z': {df['volume_z'].isna().sum()}"))
        print(bold("Sample of DataFrame before dropna:"))
        print(df[[price_col, volume_col, 'return', 'volatility', 'volume_z']].head(10))
        # Only drop rows with NaN in core features
        essential = ["return", "volatility", "volume_z"]
        n_before = len(df)
        df = df.dropna(subset=essential)
        n_after = len(df)
        print(bold(f"Rows before dropna: {n_before}, after dropna: {n_after}"))
        if n_after == 0:
            print(bold("[ERROR] All rows dropped after feature engineering. Try reducing rolling window sizes or check your data range."))
            sys.exit(1)
        if n_after < 100:
            print(bold(f"[WARN] Only {n_after} rows remaining after feature engineering. Results may be unreliable."))
        print(bold(f"Feature engineering complete. Data shape: {df.shape}"))
        print(bold(f"Final features used for ML: {ml_features[:10]} ... (total {len(ml_features)})"))
    except Exception as e:
        print(bold(f"[ERROR] Feature engineering failed: {e}"))
        sys.exit(1)

    # ========== 4. AI SIGNAL GENERATION (LightGBM) ========== #
    print_section("AI Signal Generation (LightGBM)")
    try:
        # --- Create Target Variable for ML ---
        print(bold(f"Before target creation: {len(df)} rows"))
        df["target"] = df.groupby("instrument")["return"].shift(-1)
        print(bold(f"Target NaN count: {df['target'].isna().sum()}"))
        
        # Only drop rows where target is NaN (keep the features)
        df_ml = df.dropna(subset=['target']).copy()
        print(bold(f"After target dropna: {len(df_ml)} rows"))
        
        # Prepare data for ML
        X = df_ml[ml_features].copy()
        y = df_ml["target"].copy()
        
        # Check for any remaining issues
        print(bold(f"X shape: {X.shape}, y shape: {y.shape}"))
        print(bold(f"X dtypes: {X.dtypes.value_counts()}"))
        print(bold(f"Any infinite values in X: {np.isinf(X.values).any()}"))
        print(bold(f"Any infinite values in y: {np.isinf(y.values).any()}"))
        
        # Replace infinite values with NaN and then drop
        X = X.replace([np.inf, -np.inf], np.nan)
        y = y.replace([np.inf, -np.inf], np.nan)
        
        # Debug: Check what's causing data loss
        x_na_mask = X.isna().any(axis=1)
        y_na_mask = y.isna()
        print(bold(f"X rows with NaN: {x_na_mask.sum()}"))
        print(bold(f"y rows with NaN: {y_na_mask.sum()}"))
        print(bold(f"X columns with any NaN: {X.isna().any().sum()}"))
        
        # Show which columns have NaN values
        nan_cols = X.columns[X.isna().any()].tolist()
        if nan_cols:
            print(bold(f"Columns with NaN values: {nan_cols[:10]}..."))
            for col in nan_cols[:5]:  # Show details for first 5 problematic columns
                nan_count = X[col].isna().sum()
                print(f"  {col}: {nan_count} NaNs")
        
        # Instead of dropping all rows with any NaN, let's be more selective
        # First, try to fill NaN values with median/mode
        X_cleaned = X.copy()
        for col in X_cleaned.columns:
            if X_cleaned[col].isna().any():
                if X_cleaned[col].dtype in ['float64', 'float32']:
                    X_cleaned[col] = X_cleaned[col].fillna(X_cleaned[col].median())
                else:
                    X_cleaned[col] = X_cleaned[col].fillna(X_cleaned[col].mode().iloc[0] if not X_cleaned[col].mode().empty else 0)
        
        # Only drop rows where target is NaN
        valid_mask = ~y.isna()
        X = X_cleaned[valid_mask]
        y = y[valid_mask]
        
        print(bold(f"After cleaning: X shape: {X.shape}, y shape: {y.shape}"))
        
        if len(X) == 0:
            print(bold("[ERROR] No valid data remaining after cleaning"))
            sys.exit(1)
        
        # Split data
        split_date = pd.Timestamp('2020-01-01')
        train_mask = X.index.get_level_values('datetime') < split_date
        test_mask = ~train_mask
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        
        print(bold(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}"))
        
        if len(X_train) == 0 or len(X_test) == 0:
            print(bold("[ERROR] Insufficient data for train/test split"))
            sys.exit(1)
        
        # Train LightGBM model
        model = LGBMRegressor(
            n_estimators=50,  # Reduced for faster training
            learning_rate=0.1,
            max_depth=4,  # Reduced to prevent overfitting
            random_state=42,
            verbose=-1
        )
        
        model.fit(X_train.values, y_train.values)  # Use .values to ensure 2D array
        
        # Generate predictions
        y_pred = model.predict(X_test.values)
        
        # Create signals DataFrame
        signals_df = pd.DataFrame({
            'datetime': X_test.index.get_level_values('datetime'),
            'instrument': X_test.index.get_level_values('instrument'),
            'signal': y_pred
        }).set_index(['datetime', 'instrument'])
        
        # Add signals back to main dataframe for compatibility
        df_ml = df_ml.join(signals_df, how='left')
        df_ml['signal'] = df_ml['signal'].fillna(0)  # Fill missing signals with 0
        
        print(bold(f"Generated {len(signals_df)} AI signals"))
        print(bold(f"Signal statistics: mean={signals_df['signal'].mean():.6f}, std={signals_df['signal'].std():.6f}"))
        print(bold(f"LightGBM model trained. Example signals:\n{df_ml[['signal']].head()}"))
        # --- Feature Importance Visualization ---
        import matplotlib.pyplot as plt
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:20]  # Top 20 features
        plt.figure(figsize=(10,6))
        plt.title("Top 20 Feature Importances (LightGBM)")
        plt.bar(range(len(indices)), importances[indices], align="center")
        plt.xticks(range(len(indices)), [ml_features[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "feature_importance.png"))
        plt.close()
        print(bold(f"Feature importance chart saved to {RESULTS_DIR}/feature_importance.png"))
    except Exception as e:
        print(bold(f"[ERROR] AI signal generation failed: {e}"))
        sys.exit(1)

    # ========== 5. PORTFOLIO OPTIMIZATION (PyPortfolioOpt) ========== #
    print_section("Portfolio Optimization (PyPortfolioOpt)")
    try:
        # Use last available signals for allocation
        latest_signals = df_ml.groupby("instrument").tail(1).reset_index()
        print(bold(f"Latest signals shape: {latest_signals.shape}"))
        print(bold(f"Available instruments: {latest_signals['instrument'].unique()}"))
        
        # Create price data using the selected price column
        price_data = df_ml.reset_index().pivot_table(index="datetime", columns="instrument", values=price_col)
        print(bold(f"Price data shape: {price_data.shape}"))
        print(bold(f"Price data columns: {list(price_data.columns)}"))
        
        # Calculate expected returns from signals
        mu = latest_signals.set_index("instrument")["signal"]
        print(bold(f"Expected returns (mu): {mu}"))
        
        # Calculate covariance matrix
        try:
            returns_data = price_data.pct_change().dropna()
            S = risk_models.sample_cov(returns_data)
            print(bold(f"Covariance matrix shape: {S.shape}"))
        except Exception as cov_error:
            print(bold(f"[WARN] Covariance calculation failed: {cov_error}. Using identity matrix."))
            n_assets = len(mu)
            S = pd.DataFrame(np.eye(n_assets), index=mu.index, columns=mu.index)
        
        # Portfolio optimization
        try:
            ef = EfficientFrontier(mu, S)
            weights = ef.max_sharpe()
            cleaned_weights = ef.clean_weights()
            
            # Calculate portfolio performance
            expected_annual_return, annual_volatility, sharpe_ratio = ef.portfolio_performance(verbose=False)
            print(bold(f"Expected annual return: {expected_annual_return:.2%}"))
            print(bold(f"Annual volatility: {annual_volatility:.2%}"))
            print(bold(f"Sharpe Ratio: {sharpe_ratio:.2f}"))
            print(bold(f"Optimized portfolio weights:\n{cleaned_weights}"))
        except Exception as opt_error:
            print(bold(f"[WARN] Portfolio optimization failed: {opt_error}. Using equal weights."))
            instruments = mu.index.tolist()
            equal_weight = 1.0 / len(instruments)
            cleaned_weights = {inst: equal_weight for inst in instruments}
            print(bold(f"Equal-weighted portfolio:\n{cleaned_weights}"))
            
    except Exception as e:
        print(bold(f"[ERROR] Portfolio optimization failed: {e}"))
        sys.exit(1)

    # ========== 6. BACKTESTING ========== #
    print_section("Backtesting Portfolio" )
    try:
        # Simple backtest: rebalance monthly, hold weights
        df_bt = df_ml.reset_index()
        df_bt = df_bt[df_bt["instrument"].isin(cleaned_weights.keys())]
        df_bt = df_bt.set_index(["datetime", "instrument"])
        # Calculate daily portfolio returns
        port_rets = []
        for date, group in df_bt.groupby(level=0):
            ret = 0
            for inst, w in cleaned_weights.items():
                try:
                    ret += w * group.loc[(date, inst)]["return"]
                except:
                    continue
            port_rets.append((date, ret))
        port_df = pd.DataFrame(port_rets, columns=["date", "return"]).set_index("date")
        port_df["cum_return"] = (1 + port_df["return"]).cumprod()
        port_df.to_csv(os.path.join(RESULTS_DIR, "portfolio_returns.csv"))
        print(bold(f"Backtest complete. Final cumulative return: {port_df['cum_return'].iloc[-1]:.2f}"))
    except Exception as e:
        print(bold(f"[ERROR] Backtesting failed: {e}"))
        sys.exit(1)

    # ========== 7. VISUALIZATION ========== #
    print_section("Visualizing Results")
    try:
        plt.figure(figsize=(12,6))
        plt.plot(port_df.index, port_df["cum_return"], label="Cumulative Return", color="blue", linewidth=2)
        plt.title("Portfolio Cumulative Return")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "cumulative_return.png"))
        plt.close()
        # Plot weights
        plt.figure(figsize=(10,5))
        plt.bar(cleaned_weights.keys(), cleaned_weights.values())
        plt.title("Optimized Portfolio Weights")
        plt.ylabel("Weight")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "portfolio_weights.png"))
        plt.close()
        print(bold(f"Charts saved to {RESULTS_DIR}/"))
    except Exception as e:
        print(bold(f"[ERROR] Visualization failed: {e}"))
        sys.exit(1)

    print_section("Pipeline Complete! All results are in the 'results/' directory.")