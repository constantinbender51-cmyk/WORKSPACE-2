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
df['net_direction'] = df['log_ret'].rolling(III_WINDOW).sum().abs()
df['path_length'] = df['log_ret'].abs().rolling(III_WINDOW).sum()
epsilon = 1e-8
df['iii'] = df['net_direction'] / (df['path_length'] + epsilon)

# --- B. DYNAMIC LEVERAGE BACKTEST ---
df['sma_fast'] = df['close'].rolling(SMA_FAST).mean()
df['sma_slow'] = df['close'].rolling(SMA_SLOW).mean()

df['strategy_equity'] = 1.0
df['buy_hold_equity'] = 1.0
df['leverage_used'] = 1.0 

equity = 1.0
hold_equity = 1.0
start_idx = max(SMA_SLOW, III_WINDOW)
is_busted = False

for i in range(start_idx, len(df)):
    today = df.index[i]
    prev_close = df['close'].iloc[i-1]
    prev_fast = df['sma_fast'].iloc[i-1]
    prev_slow = df['sma_slow'].iloc[i-1]
    prev_iii = df['iii'].iloc[i-1]
    
    # Dynamic Leverage Logic
    if prev_iii < 0.2:
        leverage = 1.0
    elif prev_iii < 0.6:
        leverage = 2.0
    else:
        leverage = 4.0
        
    df.at[today, 'leverage_used'] = leverage
    
    open_p = df['open'].iloc[i]
    high_p = df['high'].iloc[i]
    low_p = df['low'].iloc[i]
    close_p = df['close'].iloc[i]
    
    base_ret = 0.0
    
    if prev_close > prev_fast and prev_close > prev_slow:
        entry = open_p
        sl = entry * (1 - SL_PCT)
        tp = entry * (1 + TP_PCT)
        if low_p <= sl: base_ret = -SL_PCT
        elif high_p >= tp: base_ret = TP_PCT
        else: base_ret = (close_p - entry) / entry
        
    elif prev_close < prev_fast and prev_close < prev_slow:
        entry = open_p
        sl = entry * (1 + SL_PCT)
        tp = entry * (1 - TP_PCT)
        if high_p >= sl: base_ret = -SL_PCT
        elif low_p <= tp: base_ret = TP_PCT
        else: base_ret = (entry - close_p) / entry
    
    if not is_busted:
        daily_ret = base_ret * leverage
        equity *= (1 + daily_ret)
        if equity <= 0.05:
            equity = 0
            is_busted = True
    
    bh_ret = (close_p - df['close'].iloc[i-1]) / df['close'].iloc[i-1]
    hold_equity *= (1 + bh_ret)
    
    df.at[today, 'strategy_equity'] = equity
    df.at[today, 'buy_hold_equity'] = hold_equity

# --- C. CALCULATE METRICS ---
def get_metrics(equity_series):
    # Daily Returns
    ret = equity_series.pct_change().fillna(0)
    
    # 1. Total Return
    total_ret = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1
    
    # 2. CAGR (Annualized)
    days = (equity_series.index[-1] - equity_series.index[0]).days
    cagr = (equity_series.iloc[-1] / equity_series.iloc[0]) ** (365.0 / days) - 1
    
    # 3. Max Drawdown
    roll_max = equity_series.cummax()
    drawdown = (equity_series - roll_max) / roll_max
    max_dd = drawdown.min()
    
    # 4. Sharpe Ratio (Ann.)
    sharpe = (ret.mean() / ret.std()) * np.sqrt(365) if ret.std() != 0 else 0
    
    # 5. Calmar Ratio
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    
    return total_ret, cagr, max_dd, sharpe, calmar

# Compute for plotting range
plot_data = df.iloc[start_idx:].copy()
s_tot, s_cagr, s_mdd, s_sharpe, s_calmar = get_metrics(plot_data['strategy_equity'])
b_tot, b_cagr, b_mdd, b_sharpe, b_calmar = get_metrics(plot_data['buy_hold_equity'])

print("\n" + "="*40)
print(f"{'METRIC':<15} | {'STRATEGY':<12} | {'BUY & HOLD':<12}")
print("-" * 43)
print(f"{'Total Return':<15} | {s_tot*100:>10.1f}% | {b_tot*100:>10.1f}%")
print(f"{'CAGR':<15} | {s_cagr*100:>10.1f}% | {b_cagr*100:>10.1f}%")
print(f"{'Max Drawdown':<15} | {s_mdd*100:>10.1f}% | {b_mdd*100:>10.1f}%")
print(f"{'Sharpe Ratio':<15} | {s_sharpe:>10.2f}  | {b_sharpe:>10.2f}")
print(f"{'Calmar Ratio':<15} | {s_calmar:>10.2f}  | {b_calmar:>10.2f}")
print("="*40 + "\n")

# 3. VISUALIZATION
# ----------------
plt.figure(figsize=(14, 14))

# Plot 1: Equity
ax1 = plt.subplot(3, 1, 1)
ax1.plot(plot_data.index, plot_data['strategy_equity'], label='Dynamic Leverage Strategy', color='blue', linewidth=2)
ax1.plot(plot_data.index, plot_data['buy_hold_equity'], label='Buy & Hold', color='gray', alpha=0.5)
ax1.set_yscale('log')
ax1.set_title('Strategy Equity (Log Scale)')
ax1.legend()
ax1.grid(True, which='both', linestyle='--', alpha=0.3)

# Add Metrics Box
stats_text = (
    f"STRATEGY METRICS\n"
    f"----------------\n"
    f"CAGR:      {s_cagr*100:.1f}%\n"
    f"Max DD:    {s_mdd*100:.1f}%\n"
    f"Sharpe:    {s_sharpe:.2f}\n"
    f"Calmar:    {s_calmar:.2f}"
)
props = dict(boxstyle='round', facecolor='white', alpha=0.9)
ax1.text(0.02, 0.95, stats_text, transform=ax1.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

# Plot 2: Leverage Regimes
ax2 = plt.subplot(3, 1, 2, sharex=ax1)
ax2.step(plot_data.index, plot_data['leverage_used'], where='post', color='black', linewidth=1)
ax2.fill_between(plot_data.index, 0, plot_data['leverage_used'], step='post', alpha=0.2, color='purple')
ax2.set_yticks([1, 2, 4])
ax2.set_yticklabels(['1x', '2x', '4x'])
ax2.set_title('Leverage Deployed (Based on Yesterday\'s Efficiency)')
ax2.grid(True, axis='x', alpha=0.3)

# Plot 3: III Index
ax3 = plt.subplot(3, 1, 3, sharex=ax1)
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
ax3.axhline(0.6, color='green', linestyle='--', alpha=0.5, label='4x Threshold')
ax3.axhline(0.2, color='red', linestyle='--', alpha=0.5, label='1x Threshold')
ax3.set_ylabel('Efficiency Ratio')
ax3.set_title('Inverted Inefficiency Index (III)')
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
