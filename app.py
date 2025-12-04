import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. DATA FETCHING
# -----------------------------------------------------------------------------
def fetch_binance_data(symbol="BTCUSDT", interval="1d", start_date="2018-01-01"):
    base_url = "https://api.binance.com/api/v3/klines"
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    
    all_data = []
    limit = 1000
    current_start = start_ts
    
    print(f"Fetching {symbol} data from {start_date}...")
    
    while True:
        params = {"symbol": symbol, "interval": interval, "startTime": current_start, "limit": limit}
        try:
            response = requests.get(base_url, params=params)
            data = response.json()
            if not isinstance(data, list) or len(data) == 0: break
            all_data.extend(data)
            current_start = data[-1][6] + 1
            if len(data) < limit: break
            time.sleep(0.05)
        except Exception as e:
            print(f"Error: {e}")
            break
            
    df = pd.DataFrame(all_data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "num_trades", "taker_base_vol", "taker_quote_vol", "ignore"
    ])
    
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    df["date"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("date", inplace=True)
    
    return df[["open", "high", "low", "close"]]

# -----------------------------------------------------------------------------
# 2. GRID SEARCH (TREND + FILTER LOGIC)
# -----------------------------------------------------------------------------
def run_filter_grid(df):
    """
    Logic:
    1. Trend = Sign(Open - SMA)
    2. Filter = Flat if abs(Open/SMA - 1) < x
    3. Stop Loss s%
    """
    print("\n--- Starting Grid Search (Trend + Noise Filter) ---")
    
    # --- Parameter Spaces ---
    sma_periods = np.arange(1, 366, 2)
    x_values = np.arange(-0.05, 0.051, 0.005)
    s_values = np.arange(0.01, 0.11, 0.01)
    
    # Base Arrays
    closes = df['close'].to_numpy()
    opens = df['open'].to_numpy()
    highs = df['high'].to_numpy()
    lows = df['low'].to_numpy()
    
    # Intraday Returns (Open -> Close)
    intraday_returns = np.log(df['close'] / df['open']).to_numpy()
    # Benchmark (Close -> Close)
    benchmark_returns = np.log(df['close'] / df['close'].shift(1)).fillna(0).to_numpy()
    
    best_sharpe = -np.inf
    best_params = None
    best_curve = None
    
    # Heatmap Data
    results_matrix = np.zeros((len(sma_periods), len(x_values)))
    
    start_time = time.time()
    
    # Loop SMAs
    for i, period in enumerate(sma_periods):
        # SMA of Closes, shifted 1 (Yesterday's SMA)
        sma = df['close'].rolling(window=period).mean().shift(1).to_numpy()
        valid_mask = ~np.isnan(sma)
        
        # 1. Base Trend Signal (Long above, Short below)
        # Using np.sign: 1 if > 0, -1 if < 0, 0 if == 0
        trend_signal = np.sign(opens - sma)
        
        # Calculate deviation once
        # (Open - SMA) / SMA
        deviation = (opens - sma) / sma
        
        # Loop Threshold X
        for j, x in enumerate(x_values):
            
            # 2. Apply Filter
            # If x is positive: Flat if inside band
            # If x is negative: Condition is always False (cannot be within negative distance)
            #                   --> Logic becomes Pure SMA Trend
            is_choppy = np.abs(deviation) < x
            
            # Position is Trend Signal, but 0 where Choppy
            positions = np.where(is_choppy, 0, trend_signal)
            
            # Optimization: If positions are all 0 (rare), skip
            if np.all(positions == 0):
                results_matrix[i, j] = 0
                continue
            
            # Loop Stop Loss S
            best_s_sharpe = -np.inf
            
            for s in s_values:
                # Calculate Raw Returns
                daily_returns = positions * intraday_returns
                
                # Check Stop Loss
                # Long Stop: Low < Open * (1 - s)
                sl_long = (positions == 1) & (lows < opens * (1 - s))
                # Short Stop: High > Open * (1 + s)
                sl_short = (positions == -1) & (highs > opens * (1 + s))
                
                # Apply Penalty
                penalty = np.log(1 - s)
                daily_returns[sl_long] = penalty
                daily_returns[sl_short] = penalty
                
                # Sharpe
                active_rets = daily_returns[valid_mask]
                if len(active_rets) == 0:
                    sharpe = 0
                else:
                    mean_r = np.mean(active_rets)
                    std_r = np.std(active_rets)
                    sharpe = (mean_r / std_r) * np.sqrt(365) if std_r > 1e-9 else 0
                
                if sharpe > best_s_sharpe:
                    best_s_sharpe = sharpe
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = (period, x, s)
                    best_curve = np.cumsum(daily_returns)
            
            results_matrix[i, j] = best_s_sharpe

    print(f"Optimization complete in {time.time() - start_time:.2f}s.")
    return best_params, best_sharpe, best_curve, benchmark_returns, results_matrix, sma_periods, x_values

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    SYMBOL = "BTCUSDT"
    
    # 1. Fetch
    df = fetch_binance_data(SYMBOL)
    
    # 2. Run
    best_params, best_sharpe, best_curve, benchmark_ret, heatmap_data, smas, xs = run_filter_grid(df)
    
    best_sma, best_x, best_s = best_params
    
    print(f"\n--- RESULTS ---")
    print(f"Best SMA Period : {best_sma}")
    print(f"Best Filter X   : {best_x:.1%}")
    print(f"Best Stop Loss S: {best_s:.1%}")
    print(f"Best Sharpe Ratio: {best_sharpe:.4f}")
    
    # 3. Visualization
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2)
    
    # Plot A: Equity Curve
    ax1 = fig.add_subplot(gs[0, :])
    dates = df.index
    market_cum = np.exp(np.cumsum(benchmark_ret))
    strat_cum = np.exp(np.pad(best_curve, (0,0))) 
    
    ax1.plot(dates, market_cum, label="Buy & Hold", color='gray', alpha=0.5)
    ax1.plot(dates, strat_cum, label=f"Best Strategy (SMA {best_sma}, x={best_x:.1%})", color='blue')
    ax1.set_title(f"Equity Curve: Trend (Above/Below SMA) + Chop Filter")
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot B: Heatmap
    ax2 = fig.add_subplot(gs[1, :])
    X, Y = np.meshgrid(xs * 100, smas)
    
    # Note on Heatmap:
    # For negative X, the "Filter" is disabled (Pure SMA).
    # You will likely see horizontal bands on the left side (x < 0) where Sharpe is constant for a given SMA.
    c = ax2.pcolormesh(X, Y, heatmap_data, shading='auto', cmap='viridis')
    fig.colorbar(c, ax=ax2, label='Sharpe Ratio')
    
    ax2.plot(best_x*100, best_sma, 'r*', markersize=15, markeredgecolor='white', label='Optimal')
    
    ax2.set_title("Sharpe Ratio Landscape")
    ax2.set_xlabel("Filter Threshold X (%)")
    ax2.set_ylabel("SMA Period")
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
