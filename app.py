import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt
from flask import Flask, render_template_string
import io
import base64

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
    return df[["open", "close"]]

# -----------------------------------------------------------------------------
# 2. GRID SEARCH (SINGLE SMA)
# -----------------------------------------------------------------------------
def run_single_sma_grid(df):
    """
    Optimizes SMA Period (1-365) and X% (-5% to +5%).
    """
    print("\n--- Starting Grid Search (Single SMA) ---")
    
    # 1. Define Parameter Spaces
    # SMA: 1, 3, 5 ... 365
    sma_periods = np.arange(1, 366, 2)
    
    # X: -0.05 to +0.05 in steps of 0.005 (0.5%)
    x_values = np.arange(-0.05, 0.051, 0.005)
    
    # Base Arrays
    closes = df['close'].to_numpy()
    opens = df['open'].to_numpy()
    market_returns = np.log(df['close'] / df['close'].shift(1)).fillna(0).to_numpy()
    
    best_sharpe = -np.inf
    best_params = None
    best_curve = None
    
    # Store results for Heatmap: [SMA, X, Sharpe]
    results_matrix = np.zeros((len(sma_periods), len(x_values)))
    
    start_time = time.time()
    
    # 2. Loop over SMAs
    for i, period in enumerate(sma_periods):
        # Calculate shifted SMA (yesterday's SMA to avoid lookahead)
        sma = df['close'].rolling(window=period).mean().shift(1).to_numpy()
        
        # Valid mask (skip NaN start)
        valid_mask = ~np.isnan(sma)
        
        # 3. Loop over X thresholds
        for j, x in enumerate(x_values):
            
            # Define Bands
            upper_band = sma * (1 + x)
            lower_band = sma * (1 - x)
            
            # Vectorized Signal Logic
            # If X is negative, Bands overlap. We prioritize Long (1) over Short (-1).
            # Logic: 
            #   If Open > Upper -> Long
            #   Else If Open < Lower -> Short
            #   Else -> Flat
            
            # Initialize positions array
            positions = np.zeros_like(opens)
            
            # Apply conditions
            # Note: np.where evaluates conditions in order. Nested to simulate elif.
            # position = 1 if (open > upper) else (-1 if (open < lower) else 0)
            
            long_mask = (opens > upper_band)
            short_mask = (opens < lower_band)
            
            # Apply Short first, then overwrite with Long to give Long priority in overlap
            # Or use np.select for clarity
            conditions = [long_mask, short_mask]
            choices = [1, -1]
            positions = np.select(conditions, choices, default=0)
            
            # Calculate Returns
            strat_returns = positions * market_returns
            
            # Sharpe Calc
            active_returns = strat_returns[valid_mask]
            if len(active_returns) == 0:
                sharpe = 0
            else:
                mean_ret = np.mean(active_returns)
                std_ret = np.std(active_returns)
                sharpe = (mean_ret / std_ret) * np.sqrt(365) if std_ret > 1e-9 else 0
            
            # Store Result
            results_matrix[i, j] = sharpe
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = (period, x)
                best_curve = np.cumsum(strat_returns)

    print(f"Optimization complete in {time.time() - start_time:.2f}s.")
    return best_params, best_sharpe, best_curve, market_returns, results_matrix, sma_periods, x_values

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def create_plot(df, best_params, best_sharpe, best_curve, market_ret, heatmap_data, smas, xs):
    """Create and return a base64 encoded plot image."""
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2)
    
    best_sma, best_x = best_params
    
    # Plot A: Equity Curve
    ax1 = fig.add_subplot(gs[0, :])
    dates = df.index
    market_cum = np.exp(np.cumsum(market_ret))
    strat_cum = np.exp(np.pad(best_curve, (0,0))) # Pad if needed, but cumsum matches length usually
    
    ax1.plot(dates, market_cum, label="Buy & Hold", color='gray', alpha=0.5)
    ax1.plot(dates, strat_cum, label=f"Best Strategy (SMA {best_sma}, x={best_x:.1%})", color='green')
    ax1.set_title("Equity Curve (Log scale)")
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot B: Heatmap (Sharpe Ratio Landscape)
    ax2 = fig.add_subplot(gs[1, :])
    # Create meshgrid for pcolormesh
    X, Y = np.meshgrid(xs * 100, smas)
    
    # pcolormesh needs x, y (corners) or centers. 
    # Using simple imshow might be easier but let's try pcolormesh for correct axis labels
    c = ax2.pcolormesh(X, Y, heatmap_data, shading='auto', cmap='viridis')
    fig.colorbar(c, ax=ax2, label='Sharpe Ratio')
    
    # Mark the best point
    ax2.plot(best_x*100, best_sma, 'rx', markersize=10, markeredgewidth=2, label='Optimal')
    
    ax2.set_title("Sharpe Ratio Heatmap")
    ax2.set_xlabel("Threshold X (%)")
    ax2.set_ylabel("SMA Period")
    ax2.legend()
    
    plt.tight_layout()
    
    # Convert plot to base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return img_base64

# Create Flask app
app = Flask(__name__)

# Global variables to store results
results_data = None

@app.route('/')
def index():
    global results_data
    
    if results_data is None:
        return "Test not run yet. Please run the test first."
    
    df, best_params, best_sharpe, best_curve, market_ret, heatmap_data, smas, xs = results_data
    best_sma, best_x = best_params
    
    # Create plot
    plot_img = create_plot(df, best_params, best_sharpe, best_curve, market_ret, heatmap_data, smas, xs)
    
    # HTML template
    html_template = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Returns Over Time - SMA Strategy Backtest</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 1200px; margin: 0 auto; }
            .results { background: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
            .plot { text-align: center; }
            img { max-width: 100%; height: auto; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Returns Over Time - SMA Strategy Backtest</h1>
            
            <div class="results">
                <h2>Test Results</h2>
                <p><strong>Best SMA Period:</strong> {{ best_sma }}</p>
                <p><strong>Best Threshold X:</strong> {{ best_x_percent }}</p>
                <p><strong>Best Sharpe Ratio:</strong> {{ best_sharpe }}</p>
            </div>
            
            <div class="plot">
                <h2>Equity Curve and Sharpe Ratio Heatmap</h2>
                <img src="data:image/png;base64,{{ plot_img }}" alt="Returns Plot">
            </div>
            
            <div style="margin-top: 20px; color: #666;">
                <p>Server running on port 8080. Refresh to see updated results if test is re-run.</p>
            </div>
        </div>
    </body>
    </html>
    '''
    
    return render_template_string(html_template, 
                                 best_sma=best_sma,
                                 best_x_percent=f"{best_x:.1%}",
                                 best_sharpe=f"{best_sharpe:.4f}",
                                 plot_img=plot_img)

if __name__ == "__main__":
    SYMBOL = "BTCUSDT"
    
    print("Starting test...")
    
    # 1. Fetch
    df = fetch_binance_data(SYMBOL)
    
    # 2. Run Grid Search
    best_params, best_sharpe, best_curve, market_ret, heatmap_data, smas, xs = run_single_sma_grid(df)
    
    best_sma, best_x = best_params
    
    print(f"\n--- RESULTS ---")
    print(f"Best SMA Period: {best_sma}")
    print(f"Best Threshold X: {best_x:.1%}")
    print(f"Best Sharpe Ratio: {best_sharpe:.4f}")
    
    # Store results globally
    global results_data
    results_data = (df, best_params, best_sharpe, best_curve, market_ret, heatmap_data, smas, xs)
    
    print("\nTest complete. Starting web server on port 8080...")
    print("Open http://localhost:8080 in your browser to view the plot.")
    
    # Start Flask server
    app.run(host='0.0.0.0', port=8080, debug=False)
