import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
from datetime import datetime
from flask import Flask, send_file
import os

# 1. CONFIGURATION
# ----------------
symbol = 'BTC/USDT'
timeframe = '1d'
start_date_str = '2018-01-01 00:00:00'

# Strategy Params
SMA_FAST = 40
SMA_SLOW = 120
SL_PCT = 0.02
TP_PCT = 0.16

# Efficiency Calculation Window
III_WINDOW = 14 

def fetch_binance_history(symbol, start_str):
    print(f"Fetching data for {symbol} starting from {start_str}...")
    exchange = ccxt.binance()
    since = exchange.parse8601(start_str)
    all_ohlcv = []
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            if not ohlcv: break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            if since > exchange.milliseconds(): break
        except Exception as e:
            print(f"Error fetching: {e}")
            break
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    print(f"Fetched {len(df)} days of data.")
    return df

# 2. PREPARE DATA
# ---------------
df = fetch_binance_history(symbol, start_date_str)

# --- A. CALCULATE INVERTED INEFFICIENCY INDEX (III) ---
df['log_ret'] = np.log(df['close'] / df['close'].shift(1))

# Numerator: Absolute Net Direction over window
df['net_direction'] = df['log_ret'].rolling(III_WINDOW).sum().abs()

# Denominator: Sum of absolute individual moves (Total Path)
df['path_length'] = df['log_ret'].abs().rolling(III_WINDOW).sum()

# The Index
epsilon = 1e-8
df['iii'] = df['net_direction'] / (df['path_length'] + epsilon)

# --- B. DYNAMIC LEVERAGE BACKTEST ---
df['sma_fast'] = df['close'].rolling(SMA_FAST).mean()
df['sma_slow'] = df['close'].rolling(SMA_SLOW).mean()

df['strategy_equity'] = 1.0
df['buy_hold_equity'] = 1.0
df['leverage_used'] = 1.0 # Track for visualization

equity = 1.0
hold_equity = 1.0
start_idx = max(SMA_SLOW, III_WINDOW)

# Check for Bankruptcy
is_busted = False

for i in range(start_idx, len(df)):
    today = df.index[i]
    
    # Yesterday's Signals (Causal)
    prev_close = df['close'].iloc[i-1]
    prev_fast = df['sma_fast'].iloc[i-1]
    prev_slow = df['sma_slow'].iloc[i-1]
    
    # Yesterday's III (Determines Today's Leverage)
    prev_iii = df['iii'].iloc[i-1]
    
    # Determine Leverage based on III Tiers
    if prev_iii < 0.2:
        leverage = 1.0
    elif prev_iii < 0.6:
        leverage = 2.0
    else:
        leverage = 4.0
        
    df.at[today, 'leverage_used'] = leverage
    
    # Today's Execution
    open_p = df['open'].iloc[i]
    high_p = df['high'].iloc[i]
    low_p = df['low'].iloc[i]
    close_p = df['close'].iloc[i]
    
    # Base Strategy Return (Unleveraged)
    base_ret = 0.0
    
    # Long
    if prev_close > prev_fast and prev_close > prev_slow:
        entry = open_p
        sl = entry * (1 - SL_PCT)
        tp = entry * (1 + TP_PCT)
        if low_p <= sl: base_ret = -SL_PCT
        elif high_p >= tp: base_ret = TP_PCT
        else: base_ret = (close_p - entry) / entry
        
    # Short
    elif prev_close < prev_fast and prev_close < prev_slow:
        entry = open_p
        sl = entry * (1 + SL_PCT)
        tp = entry * (1 - TP_PCT)
        if high_p >= sl: base_ret = -SL_PCT
        elif low_p <= tp: base_ret = TP_PCT
        else: base_ret = (entry - close_p) / entry
    
    # Apply Leverage
    if not is_busted:
        daily_ret = base_ret * leverage
        equity *= (1 + daily_ret)
        
        if equity <= 0.05: # Liquidated
            equity = 0
            is_busted = True
    
    # Buy & Hold
    bh_ret = (close_p - df['close'].iloc[i-1]) / df['close'].iloc[i-1]
    hold_equity *= (1 + bh_ret)
    
    df.at[today, 'strategy_equity'] = equity
    df.at[today, 'buy_hold_equity'] = hold_equity

# 3. VISUALIZATION
# ----------------
plt.figure(figsize=(14, 14))

# Filter data for plotting
plot_data = df.iloc[start_idx:].copy()

# Plot 1: Equity Curves
ax1 = plt.subplot(3, 1, 1)
ax1.plot(plot_data.index, plot_data['strategy_equity'], label='Dynamic Leverage Strategy', color='blue', linewidth=2)
ax1.plot(plot_data.index, plot_data['buy_hold_equity'], label='Buy & Hold', color='gray', alpha=0.5)
ax1.set_yscale('log')
ax1.set_title('Strategy Equity (Dynamic Leverage: 1x/2x/4x)')
ax1.legend()
ax1.grid(True, which='both', linestyle='--', alpha=0.3)

# Plot 2: Leverage Regimes (Step Plot)
ax2 = plt.subplot(3, 1, 2, sharex=ax1)
# Create a colored step plot for leverage
# We'll use fill_between to show the regimes
ax2.step(plot_data.index, plot_data['leverage_used'], where='post', color='black', linewidth=1)
ax2.fill_between(plot_data.index, 0, plot_data['leverage_used'], step='post', alpha=0.2, color='purple')

# Add context
ax2.set_yticks([1, 2, 4])
ax2.set_yticklabels(['1x (Chop)', '2x (Normal)', '4x (Trend)'])
ax2.set_title('Leverage Deployed (Based on Yesterday\'s Efficiency)')
ax2.grid(True, axis='x', alpha=0.3)

# Plot 3: The Inverted Inefficiency Index (III)
ax3 = plt.subplot(3, 1, 3, sharex=ax1)

# Correctly handle dates for LineCollection
dates_num = mdates.date2num(plot_data.index.to_pydatetime())
iii_values = plot_data['iii'].values

points = np.array([dates_num, iii_values]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

norm = plt.Normalize(0, 1)
lc = LineCollection(segments, cmap='RdYlGn', norm=norm)
lc.set_array(iii_values)
lc.set_linewidth(1.5)
ax3.add_collection(lc)

ax3.set_xlim(dates_num.min(), dates_num.max())
ax3.set_ylim(0, 1)

# Add Threshold lines
ax3.axhline(0.6, color='green', linestyle='--', alpha=0.5, label='4x Threshold (>0.6)')
ax3.axhline(0.2, color='red', linestyle='--', alpha=0.5, label='1x Threshold (<0.2)')
ax3.set_ylabel('Efficiency Ratio (III)')
ax3.set_title('Underlying Signal: Inverted Inefficiency Index')
ax3.legend(loc='upper left')
ax3.grid(True, alpha=0.3)

plt.tight_layout()

# Save
plot_dir = '/app/static'
if not os.path.exists(plot_dir): os.makedirs(plot_dir)
plot_path = os.path.join(plot_dir, 'plot.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')

# Flask
app = Flask(__name__)
@app.route('/')
def serve_plot(): return send_file(plot_path, mimetype='image/png')
@app.route('/health')
def health(): return 'OK', 200

if __name__ == '__main__':
    print(f"Final Strategy Equity: {equity:.2f}x")
    print("\nStarting Web Server...")
    app.run(host='0.0.0.0', port=8080, debug=False)
