import pandas as pd
import numpy as np
import requests
from binance.client import Client
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from flask import Flask, send_file, jsonify
import io
import datetime
from threading import Thread
import time
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variable to store the latest data
latest_data = None
latest_covariance = None

# Binance API configuration
BINANCE_API_KEY = 'your_api_key_here'  # Optional for public data
BINANCE_API_SECRET = 'your_api_secret_here'  # Optional for public data

def get_binance_client():
    """Initialize Binance client"""
    try:
        return Client(BINANCE_API_KEY, BINANCE_API_SECRET)
    except:
        # Use without API keys for public data (with rate limits)
        return Client()

def get_most_liquid_cryptos():
    """Fetch the top 10 most liquid cryptocurrencies from Binance (excluding stablecoins)"""
    try:
        client = get_binance_client()
        
        # Get 24hr ticker data to sort by volume
        tickers = client.get_ticker()
        
        # Filter USDT pairs and sort by quote volume
        usdt_pairs = [ticker for ticker in tickers if ticker['symbol'].endswith('USDT')]
        
        # Sort by quote volume (most liquid first)
        usdt_pairs.sort(key=lambda x: float(x['quoteVolume']), reverse=True)
        
        # Filter out stablecoins and get top 10
        stablecoin_keywords = ['BUSD', 'USDC', 'DAI', 'TUSD', 'PAX', 'USDP']
        liquid_cryptos = []
        
        for pair in usdt_pairs:
            symbol = pair['symbol'].replace('USDT', '')
            
            # Skip stablecoins
            if any(stable in symbol for stable in stablecoin_keywords):
                continue
            
            # Skip if already added
            if any(crypto['symbol'] == symbol for crypto in liquid_cryptos):
                continue
            
            liquid_cryptos.append({
                'symbol': symbol,
                'base_asset': symbol,
                'quote_asset': 'USDT',
                'full_symbol': pair['symbol'],
                'volume': float(pair['quoteVolume'])
            })
            
            if len(liquid_cryptos) >= 10:
                break
        
        # Add Bitcoin as the first asset if not already included
        btc_exists = any(crypto['symbol'] == 'BTC' for crypto in liquid_cryptos)
        if not btc_exists:
            # Find BTC pair
            btc_pair = next((p for p in usdt_pairs if p['symbol'] == 'BTCUSDT'), None)
            if btc_pair:
                liquid_cryptos.insert(0, {
                    'symbol': 'BTC',
                    'base_asset': 'BTC',
                    'quote_asset': 'USDT',
                    'full_symbol': 'BTCUSDT',
                    'volume': float(btc_pair['quoteVolume'])
                })
        
        return liquid_cryptos[:11]  # Ensure we have exactly 11 assets (BTC + 10 others)
    
    except Exception as e:
        print(f"Error fetching liquid cryptos from Binance: {e}")
        # Fallback list of major cryptocurrencies
        return [
            {'symbol': 'BTC', 'base_asset': 'BTC', 'quote_asset': 'USDT', 'full_symbol': 'BTCUSDT'},
            {'symbol': 'ETH', 'base_asset': 'ETH', 'quote_asset': 'USDT', 'full_symbol': 'ETHUSDT'},
            {'symbol': 'BNB', 'base_asset': 'BNB', 'quote_asset': 'USDT', 'full_symbol': 'BNBUSDT'},
            {'symbol': 'XRP', 'base_asset': 'XRP', 'quote_asset': 'USDT', 'full_symbol': 'XRPUSDT'},
            {'symbol': 'ADA', 'base_asset': 'ADA', 'quote_asset': 'USDT', 'full_symbol': 'ADAUSDT'},
            {'symbol': 'SOL', 'base_asset': 'SOL', 'quote_asset': 'USDT', 'full_symbol': 'SOLUSDT'},
            {'symbol': 'DOT', 'base_asset': 'DOT', 'quote_asset': 'USDT', 'full_symbol': 'DOTUSDT'},
            {'symbol': 'DOGE', 'base_asset': 'DOGE', 'quote_asset': 'USDT', 'full_symbol': 'DOGEUSDT'},
            {'symbol': 'AVAX', 'base_asset': 'AVAX', 'quote_asset': 'USDT', 'full_symbol': 'AVAXUSDT'},
            {'symbol': 'LINK', 'base_asset': 'LINK', 'quote_asset': 'USDT', 'full_symbol': 'LINKUSDT'},
            {'symbol': 'LTC', 'base_asset': 'LTC', 'quote_asset': 'USDT', 'full_symbol': 'LTCUSDT'}
        ]

def fetch_binance_historical_data(symbol, start_date, end_date):
    """Fetch historical daily price data from Binance"""
    try:
        client = get_binance_client()
        
        # Convert dates to strings for Binance API
        start_str = start_date.strftime("%d %b, %Y")
        end_str = end_date.strftime("%d %b, %Y")
        
        print(f"Fetching data for {symbol} from {start_str} to {end_str}")
        
        # Get historical klines (candlestick data)
        klines = client.get_historical_klines(
            symbol=symbol,
            interval=Client.KLINE_INTERVAL_1DAY,
            start_str=start_str,
            end_str=end_str
        )
        
        if not klines:
            print(f"No data returned for {symbol}")
            return None
        
        # Parse klines data
        dates = []
        prices = []
        
        for kline in klines:
            timestamp = datetime.datetime.fromtimestamp(kline[0] / 1000)
            close_price = float(kline[4])  # Close price
            dates.append(timestamp)
            prices.append(close_price)
        
        df = pd.DataFrame({
            'date': dates,
            'price': prices
        }).set_index('date')
        
        print(f"Fetched {len(df)} days of data for {symbol}")
        return df
    
    except Exception as e:
        print(f"Error fetching Binance data for {symbol}: {e}")
        return None

def calculate_relative_returns_and_covariance(cryptos_data):
    """Calculate relative price changes and covariance with Bitcoin"""
    # Combine all prices into a single DataFrame
    combined_data = pd.DataFrame()
    
    for symbol, data in cryptos_data.items():
        combined_data[symbol] = data['price']
    
    # Calculate daily returns
    returns_data = combined_data.pct_change().dropna()
    
    # Calculate covariance with Bitcoin
    btc_returns = returns_data['BTC']
    covariance_data = {}
    
    for symbol in returns_data.columns:
        if symbol != 'BTC':
            cov = returns_data[symbol].cov(btc_returns)
            correlation = returns_data[symbol].corr(btc_returns)
            beta = cov / btc_returns.var()  # Beta coefficient
            
            covariance_data[symbol] = {
                'covariance': cov,
                'correlation': correlation,
                'beta': beta,
                'volatility': returns_data[symbol].std()
            }
    
    # Also add BTC stats
    covariance_data['BTC'] = {
        'covariance': 0,  # BTC with itself
        'correlation': 1,
        'beta': 1,
        'volatility': btc_returns.std()
    }
    
    return returns_data, covariance_data

def create_plot(returns_data, cryptos_info):
    """Create a matplotlib plot of relative price changes"""
    plt.style.use('dark_background')
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))
    
    # Plot 1: Cumulative returns
    cumulative_returns = (1 + returns_data).cumprod()
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(cumulative_returns.columns)))
    color_map = {symbol: color for symbol, color in zip(cumulative_returns.columns, colors)}
    
    for symbol in cumulative_returns.columns:
        if symbol != 'BTC':
            ax1.plot(cumulative_returns.index, cumulative_returns[symbol], 
                    label=f'{symbol}', linewidth=2, color=color_map[symbol])
    
    # Plot Bitcoin separately for emphasis
    ax1.plot(cumulative_returns.index, cumulative_returns['BTC'], 
            label='BTC', linewidth=3, color='orange', linestyle='--')
    
    ax1.set_title('Binance - Cumulative Relative Price Changes (Jan 2022 - Sep 2023)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax1.set_ylabel('Cumulative Returns', fontsize=12)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Daily returns heatmap-style
    recent_returns = returns_data.tail(30)  # Last 30 days
    symbols = [col for col in recent_returns.columns if col != 'BTC']
    
    for symbol in symbols:
        ax2.plot(recent_returns.index, recent_returns[symbol] * 100, 
                label=f'{symbol}', alpha=0.7, linewidth=1.5, color=color_map[symbol])
    
    ax2.plot(recent_returns.index, recent_returns['BTC'] * 100, 
            label='BTC', linewidth=2, color='orange')
    
    ax2.set_title('Recent Daily Returns (%) - Last 30 Days', 
                 fontsize=14, fontweight='bold', pad=20)
    ax2.set_ylabel('Daily Return (%)', fontsize=12)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Rolling correlation with BTC (30-day window)
    rolling_correlation = pd.DataFrame()
    for symbol in returns_data.columns:
        if symbol != 'BTC':
            rolling_corr = returns_data[symbol].rolling(window=30).corr(returns_data['BTC'])
            rolling_correlation[symbol] = rolling_corr
    
    for symbol in rolling_correlation.columns:
        ax3.plot(rolling_correlation.index, rolling_correlation[symbol], 
                label=f'{symbol}', alpha=0.7, linewidth=1.5, color=color_map[symbol])
    
    ax3.set_title('30-Day Rolling Correlation with Bitcoin', 
                 fontsize=14, fontweight='bold', pad=20)
    ax3.set_ylabel('Correlation', fontsize=12)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save plot to bytes
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png', dpi=150, bbox_inches='tight')
    img_bytes.seek(0)
    plt.close()
    
    return img_bytes

def create_covariance_table(covariance_data):
    """Create a table visualization of covariance metrics"""
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data for table
    symbols = [sym for sym in covariance_data.keys() if sym != 'BTC']
    metrics = ['Covariance', 'Correlation', 'Beta', 'Volatility']
    
    data = []
    for symbol in symbols:
        data.append([
            f"{covariance_data[symbol]['covariance']:.6f}",
            f"{covariance_data[symbol]['correlation']:.4f}",
            f"{covariance_data[symbol]['beta']:.4f}",
            f"{covariance_data[symbol]['volatility']:.4f}"
        ])
    
    # Create table
    table = ax.table(
        cellText=data,
        rowLabels=symbols,
        colLabels=metrics,
        cellLoc='center',
        loc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    ax.set_title('Covariance Metrics with Bitcoin', fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png', dpi=150, bbox_inches='tight')
    img_bytes.seek(0)
    plt.close()
    
    return img_bytes

def update_data():
    """Background task to update crypto data periodically"""
    global latest_data, latest_covariance
    
    while True:
        try:
            print("Updating crypto data from Binance...")
            
            # Define date range
            start_date = datetime.datetime(2022, 1, 1)
            end_date = datetime.datetime(2023, 9, 30)
            
            # Get most liquid cryptocurrencies
            cryptos = get_most_liquid_cryptos()
            print(f"Fetching data for: {[crypto['symbol'] for crypto in cryptos]}")
            
            # Fetch data for all cryptocurrencies
            cryptos_data = {}
            successful_fetches = 0
            
            for crypto in cryptos:
                print(f"Fetching {crypto['symbol']} ({crypto['full_symbol']})...")
                data = fetch_binance_historical_data(crypto['full_symbol'], start_date, end_date)
                if data is not None and not data.empty:
                    cryptos_data[crypto['symbol']] = data
                    successful_fetches += 1
                    print(f"✓ Successfully fetched {len(data)} records for {crypto['symbol']}")
                else:
                    print(f"✗ Failed to fetch data for {crypto['symbol']}")
                
                time.sleep(0.5)  # Rate limiting for Binance API
            
            if successful_fetches > 1:
                # Calculate returns and covariance
                returns_data, covariance_data = calculate_relative_returns_and_covariance(cryptos_data)
                
                latest_data = {
                    'returns': returns_data,
                    'cryptos_info': cryptos,
                    'covariance_data': covariance_data,
                    'last_update': datetime.datetime.now()
                }
                latest_covariance = covariance_data
                
                print(f"Data update completed! Successfully processed {successful_fetches} assets")
                print("Covariance summary:")
                for symbol, stats in covariance_data.items():
                    if symbol != 'BTC':
                        print(f"  {symbol}: Corr={stats['correlation']:.4f}, Beta={stats['beta']:.4f}")
                
            else:
                print("Insufficient data fetched - need at least 2 assets")
                
        except Exception as e:
            print(f"Error updating data: {e}")
            import traceback
            traceback.print_exc()
        
        # Update every 2 hours (Binance has more frequent data updates)
        print("Waiting 2 hours for next update...")
        time.sleep(7200)

@app.route('/')
def index():
    """Main page with the graph"""
    global latest_data
    
    if latest_data is None:
        return """
        <html>
            <head><title>Binance Crypto Analysis</title>
            <style>
                body { background-color: #1e1e1e; color: white; font-family: Arial; padding: 20px; }
                .container { max-width: 1200px; margin: 0 auto; text-align: center; }
                .loading { color: #4CAF50; font-size: 18px; margin-top: 50px; }
            </style>
            </head>
            <body>
                <div class="container">
                    <h1>📊 Binance Cryptocurrency Analysis</h1>
                    <p>Analyzing the 10 most liquid crypto assets vs Bitcoin (Jan 2022 - Sep 2023)</p>
                    <div class="loading">
                        <p>Data is being loaded from Binance API...</p>
                        <p>This may take a few minutes for the initial load.</p>
                    </div>
                </div>
            </body>
        </html>
        """
    
    # Create and return the plot
    img_bytes = create_plot(latest_data['returns'], latest_data['cryptos_info'])
    return send_file(img_bytes, mimetype='image/png')

@app.route('/table')
def covariance_table():
    """Page showing covariance metrics as a table"""
    global latest_covariance
    
    if latest_covariance is None:
        return "Data not available yet", 503
    
    img_bytes = create_covariance_table(latest_covariance)
    return send_file(img_bytes, mimetype='image/png')

@app.route('/data')
def get_data():
    """API endpoint to get covariance data"""
    global latest_covariance
    
    if latest_covariance is None:
        return jsonify({"error": "Data not available yet"}), 503
    
    return jsonify({
        "covariance_data": latest_covariance,
        "last_updated": datetime.datetime.now().isoformat(),
        "data_source": "Binance"
    })

@app.route('/status')
def status():
    """Status page"""
    global latest_data
    
    status_info = {
        "status": "running",
        "data_source": "Binance API",
        "last_update": datetime.datetime.now().isoformat(),
        "assets_loaded": len(latest_data['cryptos_info']) if latest_data else 0
    }
    
    html = f"""
    <html>
        <head>
            <title>Status - Binance Crypto Analysis</title>
            <style>
                body {{ background-color: #1e1e1e; color: white; font-family: Arial; padding: 20px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .card {{ background-color: #2d2d2d; padding: 20px; margin: 10px 0; border-radius: 8px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #444; }}
                th {{ background-color: #3d3d3d; }}
                a {{ color: #4CAF50; text-decoration: none; }}
                a:hover {{ text-decoration: underline; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🔍 Binance Cryptocurrency Analysis - Status</h1>
                
                <div class="card">
                    <h2>System Status</h2>
                    <p><strong>Status:</strong> <span style="color: #4CAF50;">{status_info['status'].upper()}</span></p>
                    <p><strong>Data Source:</strong> {status_info['data_source']}</p>
                    <p><strong>Last Update:</strong> {status_info['last_update']}</p>
                    <p><strong>Assets Loaded:</strong> {status_info['assets_loaded']}</p>
                </div>
    """
    
    if latest_data and latest_covariance:
        html += """
                <div class="card">
                    <h2>Covariance with Bitcoin</h2>
                    <table>
                        <tr>
                            <th>Asset</th>
                            <th>Covariance</th>
                            <th>Correlation</th>
                            <th>Beta</th>
                            <th>Volatility</th>
                        </tr>
        """
        
        for symbol, data in latest_covariance.items():
            if symbol != 'BTC':
                corr_color = "#4CAF50" if data['correlation'] > 0.7 else "#FF9800" if data['correlation'] > 0.3 else "#f44336"
                html += f"""
                        <tr>
                            <td><strong>{symbol}</strong></td>
                            <td>{data['covariance']:.6f}</td>
                            <td style="color: {corr_color}">{data['correlation']:.4f}</td>
                            <td>{data['beta']:.4f}</td>
                            <td>{data['volatility']:.4f}</td>
                        </tr>
                """
        
        html += """
                    </table>
                </div>
        """
    
    html += """
                <div class="card">
                    <h2>Navigation</h2>
                    <p>
                        <a href="/">📈 View Main Graph</a> | 
                        <a href="/table">📊 View Covariance Table</a> | 
                        <a href="/data">🔗 Raw Data (JSON)</a>
                    </p>
                </div>
            </div>
        </body>
    </html>
    """
    
    return html

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": datetime.datetime.now().isoformat()})

if __name__ == '__main__':
    # Start background data update thread
    update_thread = Thread(target=update_data, daemon=True)
    update_thread.start()
    
    # Wait a bit for initial data load
    print("Starting Binance Crypto Analysis Server...")
    print("Initial data loading may take a few minutes.")
    print("Server will be available at http://0.0.0.0:8080")
    time.sleep(2)
    
    # Start Flask server
    app.run(host='0.0.0.0', port=8080, debug=False)
