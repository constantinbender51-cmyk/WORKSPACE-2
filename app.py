import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template

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
# 2. GRID SEARCH (INTRADAY OPEN-TO-CLOSE)
# -----------------------------------------------------------------------------
def run_single_sma_grid(df):
    """
    Optimizes Intraday Strategy:
    1. SMA Period (1-365)
    2. Entry Threshold X% (-5% to +5%)
    3. Stop Loss S% (1% to 10%)
    
    Simulates: Enter Open -> Exit Close (Daily Compounding).
    """
    print("\n--- Starting Grid Search (Intraday: Open-to-Close) ---")
    
    # --- Parameter Spaces ---
    sma_periods = np.arange(1, 366, 2)
    x_values = np.arange(-0.05, 0.051, 0.005)
    s_values = np.arange(0.01, 0.11, 0.01)
    
    # Base Arrays
    closes = df['close'].to_numpy()
    opens = df['open'].to_numpy()
    highs = df['high'].to_numpy()
    lows = df['low'].to_numpy()
    
    # --- RETURN CALCULATIONS ---
    # Benchmark: Standard Close-to-Close (Holding overnight)
    benchmark_returns = np.log(df['close'] / df['close'].shift(1)).fillna(0).to_numpy()
    
    # Strategy Basis: Intraday Open-to-Close
    # Logic: We enter at Open, Exit at Close. No overnight gap exposure.
    intraday_returns = np.log(df['close'] / df['open']).to_numpy()
    
    best_sharpe = -np.inf
    best_params = None
    best_curve = None
    
    results_matrix = np.zeros((len(sma_periods), len(x_values)))
    
    start_time = time.time()
    
    # 2. Loop over SMAs
    for i, period in enumerate(sma_periods):
        # SMA calculated on Closes, shifted 1 day to represent "Yesterday's SMA"
        sma = df['close'].rolling(window=period).mean().shift(1).to_numpy()
        valid_mask = ~np.isnan(sma)
        
        # 3. Loop over Entry Threshold X
        for j, x in enumerate(x_values):
            
            upper_band = sma * (1 + x)
            lower_band = sma * (1 - x)
            
            # --- Determine Position at Open ---
            # Compare Open price to the Bands derived from Yesterday's SMA
            long_entry = (opens > upper_band)
            short_entry = (opens < lower_band)
            
            # Resolve overlaps (Long priority if X is negative)
            raw_positions = np.zeros_like(opens)
            raw_positions[short_entry] = -1
            raw_positions[long_entry] = 1
            
            # --- Inner Loop: Stop Loss S ---
            best_s_sharpe = -np.inf
            
            for s in s_values:
                # 1. Calculate Base Strategy Returns (Intraday)
                daily_returns = raw_positions * intraday_returns
                
                # 2. Check Stop Loss Conditions (Intraday High/Low vs Open)
                sl_long_hit = (raw_positions == 1) & (lows < opens * (1 - s))
                sl_short_hit = (raw_positions == -1) & (highs > opens * (1 + s))
                
                # 3. Apply Penalty (Realized loss of s%)
                penalty = np.log(1 - s)
                daily_returns[sl_long_hit] = penalty
                daily_returns[sl_short_hit] = penalty
                
                # 4. Calculate Sharpe
                active_rets = daily_returns[valid_mask]
                if len(active_rets) == 0:
                    sharpe = 0
                else:
                    mean_r = np.mean(active_rets)
                    std_r = np.std(active_rets)
                    # Annualize Sharpe (365 trading days)
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
# 3. WEB SERVER
# -----------------------------------------------------------------------------
app = Flask(__name__)

@app.route('/')
def index():
    SYMBOL = "BTCUSDT"
    
    # 1. Fetch data
    df = fetch_binance_data(SYMBOL)
    
    # 2. Run Grid Search
    best_params, best_sharpe, best_curve, benchmark_ret, heatmap_data, smas, xs = run_single_sma_grid(df)
    
    best_sma, best_x, best_s = best_params
    
    print(f"\n--- RESULTS (Intraday Open-to-Close) ---")
    print(f"Best SMA Period : {best_sma}")
    print(f"Best Threshold X: {best_x:.1%}")
    print(f"Best Stop Loss S: {best_s:.1%}")
    print(f"Best Sharpe Ratio: {best_sharpe:.4f}")
    
    # 3. Create visualization
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2)
    
    # Plot A: Equity Curve with Price and SMA on Secondary Axis
    ax1 = fig.add_subplot(gs[0, :])
    dates = df.index
    
    # Compare against standard Buy & Hold (Close-to-Close)
    market_cum = np.exp(np.cumsum(benchmark_ret))
    
    # Strategy Curve (Intraday Compounded)
    strat_cum = np.exp(np.pad(best_curve, (0,0))) 
    
    # Plot equity curves on left axis
    ax1.plot(dates, market_cum, label="Buy & Hold (Standard)", color='gray', alpha=0.5)
    ax1.plot(dates, strat_cum, label=f"Intraday Strategy (SMA {best_sma}, x={best_x:.1%})", color='purple')
    ax1.set_title(f"Equity Curve with Price and SMA (SMA {best_sma})")
    ax1.set_yscale('linear')
    ax1.set_ylabel('Equity (Log Scale)', color='black')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Create secondary axis for price and SMA
    ax1_right = ax1.twinx()
    
    # Calculate SMA for the best period
    sma_series = df['close'].rolling(window=best_sma).mean()
    
    # Plot price and SMA on right axis
    ax1_right.plot(dates, df['close'], label='BTC Price', color='blue', alpha=0.3, linewidth=0.8)
    ax1_right.plot(dates, sma_series, label=f'SMA {best_sma}', color='red', alpha=0.7, linewidth=1.2)
    ax1_right.set_ylabel('Price (USD)', color='black')
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_right.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(0, 1), bbox_transform=ax1.transAxes)
    
    # Plot B: Heatmap
    ax2 = fig.add_subplot(gs[1, :])
    X, Y = np.meshgrid(xs * 100, smas)
    
    c = ax2.pcolormesh(X, Y, heatmap_data, shading='auto', cmap='viridis')
    fig.colorbar(c, ax=ax2, label='Sharpe Ratio')
    
    ax2.plot(best_x*100, best_sma, 'r*', markersize=15, markeredgecolor='white', label='Optimal')
    ax2.set_title("Sharpe Ratio Heatmap (Best Stop Loss per cell)")
    ax2.set_xlabel("Threshold X (%)")
    ax2.set_ylabel("SMA Period")
    ax2.legend()
    
    plt.tight_layout()
    
    # Convert plot to base64 for HTML display
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close(fig)
    
    # Prepare data for template
    data = {
        'symbol': SYMBOL,
        'best_sma': best_sma,
        'best_x': best_x,
        'best_s': best_s,
        'best_sharpe': best_sharpe,
        'plot_url': plot_url
    }
    
    return render_template('index.html', data=data)

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Starting web server on port 8080...")
    print("Open http://localhost:8080 in your browser to view results")
    app.run(host='0.0.0.0', port=8080, debug=False)
