import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
from flask import Flask, send_file
import io
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

def fetch_binance_data():
    """Fetch daily OHLCV data from Binance for BTC/USDT from 2018 to present"""
    print("Fetching data from Binance...")
    
    url = "https://api.binance.com/api/v3/klines"
    symbol = "BTCUSDT"
    interval = "1d"
    
    # Start from January 1, 2018
    start_date = datetime(2018, 1, 1)
    end_date = datetime.now()
    
    all_data = []
    current_start = start_date
    
    while current_start < end_date:
        start_ms = int(current_start.timestamp() * 1000)
        
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_ms,
            'limit': 1000
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if not data:
            break
            
        all_data.extend(data)
        
        # Move to next batch
        last_timestamp = data[-1][0]
        current_start = datetime.fromtimestamp(last_timestamp / 1000) + timedelta(days=1)
        
        if len(data) < 1000:
            break
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
    df = df.set_index('timestamp')
    
    print(f"Fetched {len(df)} days of data from {df.index[0]} to {df.index[-1]}")
    
    return df

def calculate_strategy(df):
    """Calculate trading signals and returns"""
    
    # Calculate SMAs
    df['sma_365'] = df['open'].rolling(window=365).mean()
    df['sma_120'] = df['open'].rolling(window=120).mean()
    
    # Generate positions
    df['position'] = 0
    
    # Long: open > SMA365 AND open > SMA120
    long_condition = (df['open'] > df['sma_365']) & (df['open'] > df['sma_120'])
    df.loc[long_condition, 'position'] = 1
    
    # Short: open < SMA365 AND open < SMA120
    short_condition = (df['open'] < df['sma_365']) & (df['open'] < df['sma_120'])
    df.loc[short_condition, 'position'] = -1
    
    # Calculate daily returns
    df['daily_return'] = (df['close'] - df['open']) / df['open']
    
    # Apply stop loss
    df['stop_loss_triggered'] = False
    
    # Long stop loss: low < 0.95 * open
    long_stop = (df['position'] == 1) & (df['low'] < 0.95 * df['open'])
    df.loc[long_stop, 'stop_loss_triggered'] = True
    
    # Short stop loss: high > 1.05 * open
    short_stop = (df['position'] == -1) & (df['high'] > 1.05 * df['open'])
    df.loc[short_stop, 'stop_loss_triggered'] = True
    
    # Calculate strategy returns
    df['strategy_return'] = df['position'] * df['daily_return']
    
    # Apply stop loss: set return to -5%
    df.loc[df['stop_loss_triggered'], 'strategy_return'] = -0.05
    
    # Apply 5x leverage
    df['strategy_return'] = df['strategy_return'] * 5
    
    # Calculate compounded returns
    df['equity'] = (1 + df['strategy_return']).cumprod()
    
    # Calculate weekly and monthly returns
    df['week'] = df.index.to_period('W')
    df['month'] = df.index.to_period('M')
    
    weekly_returns = df.groupby('week')['strategy_return'].apply(
        lambda x: (1 + x).prod() - 1
    )
    
    monthly_returns = df.groupby('month')['strategy_return'].apply(
        lambda x: (1 + x).prod() - 1
    )
    
    return df, weekly_returns, monthly_returns

def calculate_sharpe_ratio(returns, periods_per_year=252):
    """Calculate annualized Sharpe ratio"""
    if len(returns) == 0 or returns.std() == 0:
        return 0
    
    mean_return = returns.mean()
    std_return = returns.std()
    sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)
    
    return sharpe

def create_plots(df, weekly_returns, monthly_returns):
    """Create matplotlib plots"""
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Plot 1: Equity Curve
    axes[0].plot(df.index, df['equity'], linewidth=2, color='blue')
    axes[0].set_title('Equity Curve (5x Leveraged, Stop Loss at 5%)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Equity (Starting at 1.0)', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlabel('Date', fontsize=12)
    
    # Add final equity value
    final_equity = df['equity'].iloc[-1]
    axes[0].text(0.02, 0.98, f'Final Equity: {final_equity:.2f}x', 
                transform=axes[0].transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Weekly Returns
    weekly_returns_plot = weekly_returns.reset_index()
    weekly_returns_plot['week_start'] = weekly_returns_plot['week'].dt.start_time
    
    colors = ['green' if x >= 0 else 'red' for x in weekly_returns_plot[0]]
    axes[1].bar(weekly_returns_plot['week_start'], weekly_returns_plot[0], 
               width=5, color=colors, alpha=0.7)
    axes[1].set_title('Weekly Returns', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Return', fontsize=12)
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_xlabel('Date', fontsize=12)
    
    # Plot 3: Monthly Returns
    monthly_returns_plot = monthly_returns.reset_index()
    monthly_returns_plot['month_start'] = monthly_returns_plot['month'].dt.start_time
    
    colors = ['green' if x >= 0 else 'red' for x in monthly_returns_plot[0]]
    axes[2].bar(monthly_returns_plot['month_start'], monthly_returns_plot[0], 
               width=20, color=colors, alpha=0.7)
    axes[2].set_title('Monthly Returns', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Return', fontsize=12)
    axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[2].grid(True, alpha=0.3, axis='y')
    axes[2].set_xlabel('Date', fontsize=12)
    
    plt.tight_layout()
    
    # Calculate Sharpe Ratio
    sharpe = calculate_sharpe_ratio(df['strategy_return'].dropna())
    
    # Add Sharpe ratio to the plot
    fig.text(0.99, 0.01, f'Sharpe Ratio: {sharpe:.2f}', 
            ha='right', fontsize=11, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    return fig, sharpe

# Global variables to store data
data_cache = None
fig_cache = None

def initialize_data():
    """Initialize data and calculations"""
    global data_cache, fig_cache
    
    if data_cache is None:
        df = fetch_binance_data()
        df, weekly_returns, monthly_returns = calculate_strategy(df)
        fig, sharpe = create_plots(df, weekly_returns, monthly_returns)
        
        data_cache = {
            'df': df,
            'weekly_returns': weekly_returns,
            'monthly_returns': monthly_returns,
            'sharpe': sharpe
        }
        fig_cache = fig
        
        # Print summary statistics
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        print(f"Period: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        print(f"Total Days: {len(df)}")
        print(f"Final Equity: {df['equity'].iloc[-1]:.2f}x")
        print(f"Total Return: {(df['equity'].iloc[-1] - 1) * 100:.2f}%")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Max Drawdown: {((df['equity'].cummax() - df['equity']) / df['equity'].cummax()).max() * 100:.2f}%")
        print(f"Number of Trades: {(df['position'] != 0).sum()}")
        print(f"Win Rate: {(df[df['position'] != 0]['strategy_return'] > 0).mean() * 100:.2f}%")
        print("="*60 + "\n")

@app.route('/')
def index():
    """Serve the plot"""
    initialize_data()
    
    # Save figure to bytes buffer
    buf = io.BytesIO()
    fig_cache.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    
    return send_file(buf, mimetype='image/png')

@app.route('/stats')
def stats():
    """Display statistics as text"""
    initialize_data()
    
    df = data_cache['df']
    sharpe = data_cache['sharpe']
    
    stats_html = f"""
    <html>
    <head>
        <title>BTC/USD Backtest Statistics</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
            .container {{ background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            h1 {{ color: #333; }}
            .stat {{ margin: 15px 0; font-size: 16px; }}
            .label {{ font-weight: bold; color: #555; }}
            .value {{ color: #007bff; }}
            a {{ display: inline-block; margin-top: 20px; padding: 10px 20px; background-color: #007bff; color: white; text-decoration: none; border-radius: 5px; }}
            a:hover {{ background-color: #0056b3; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>BTC/USD Trading Backtest Results</h1>
            <div class="stat"><span class="label">Period:</span> <span class="value">{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}</span></div>
            <div class="stat"><span class="label">Total Days:</span> <span class="value">{len(df)}</span></div>
            <div class="stat"><span class="label">Final Equity:</span> <span class="value">{df['equity'].iloc[-1]:.2f}x</span></div>
            <div class="stat"><span class="label">Total Return:</span> <span class="value">{(df['equity'].iloc[-1] - 1) * 100:.2f}%</span></div>
            <div class="stat"><span class="label">Sharpe Ratio:</span> <span class="value">{sharpe:.2f}</span></div>
            <div class="stat"><span class="label">Max Drawdown:</span> <span class="value">{((df['equity'].cummax() - df['equity']) / df['equity'].cummax()).max() * 100:.2f}%</span></div>
            <div class="stat"><span class="label">Number of Trades:</span> <span class="value">{(df['position'] != 0).sum()}</span></div>
            <div class="stat"><span class="label">Win Rate:</span> <span class="value">{(df[df['position'] != 0]['strategy_return'] > 0).mean() * 100:.2f}%</span></div>
            <a href="/">View Charts</a>
        </div>
    </body>
    </html>
    """
    
    return stats_html

if __name__ == '__main__':
    print("Starting BTC/USD Backtest Server...")
    print("Initializing data and calculations...")
    initialize_data()
    print(f"\nServer running at http://localhost:8080")
    print("Press Ctrl+C to stop the server")
    app.run(host='0.0.0.0', port=8080, debug=False)
