import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import requests
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Required for headless environments
import io
import base64
from flask import Flask, render_template_string
import threading
import time

app = Flask(__name__)

class CryptoPredictor:
    def __init__(self):
        self.crypto_data = {}
        self.models = {}
        self.predictions = {}
        
    def fetch_crypto_data(self, symbol, start_date, end_date):
        """Fetch daily candle data from Binance API"""
        base_url = "https://api.binance.com/api/v3/klines"
        
        # Convert dates to milliseconds
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
        
        all_data = []
        current_start = start_ts
        
        while current_start < end_ts:
            params = {
                'symbol': symbol,
                'interval': '1d',
                'limit': 1000,
                'startTime': current_start,
                'endTime': end_ts
            }
            
            try:
                response = requests.get(base_url, params=params)
                data = response.json()
                
                if not data:
                    break
                    
                for candle in data:
                    all_data.append({
                        'timestamp': candle[0],
                        'open': float(candle[1]),
                        'high': float(candle[2]),
                        'low': float(candle[3]),
                        'close': float(candle[4]),
                        'volume': float(candle[5])
                    })
                
                # Move to next time period
                current_start = data[-1][0] + 86400000  # Add one day in milliseconds
                
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
                break
        
        return pd.DataFrame(all_data)
    
    def prepare_features(self, df):
        """Prepare features for machine learning"""
        df = df.copy()
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.sort_values('date')
        
        # Create features
        df['days'] = (df['date'] - df['date'].min()).dt.days
        df['price_change'] = df['close'].pct_change()
        df['volume_change'] = df['volume'].pct_change()
        df['high_low_ratio'] = df['high'] / df['low']
        
        # Rolling features
        df['ma_7'] = df['close'].rolling(window=7).mean()
        df['ma_30'] = df['close'].rolling(window=30).mean()
        df['volatility'] = df['close'].rolling(window=7).std()
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    
    def train_model(self, symbol, df):
        """Train linear regression model"""
        # Features for training
        feature_columns = ['days', 'ma_7', 'ma_30', 'volatility', 'high_low_ratio']
        
        X = df[feature_columns]
        y = df['close']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions on test set
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"{symbol} Model Performance:")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"R² Score: {r2:.4f}")
        
        return model, feature_columns, mse, r2
    
    def predict_future(self, model, features, df, days_to_predict=30):
        """Predict future prices"""
        last_row = df.iloc[-1]
        future_predictions = []
        
        for day in range(1, days_to_predict + 1):
            future_day = last_row['days'] + day
            
            # Create future feature row (simplified - in practice you'd need to update all features)
            future_features = [
                future_day,
                last_row['ma_7'],  # This would need proper updating in a real scenario
                last_row['ma_30'],
                last_row['volatility'],
                last_row['high_low_ratio']
            ]
            
            prediction = model.predict([future_features])[0]
            future_predictions.append({
                'days': future_day,
                'predicted_price': prediction,
                'date': last_row['date'] + timedelta(days=day)
            })
        
        return pd.DataFrame(future_predictions)
    
    def create_plot(self, symbol, historical_df, predictions_df):
        """Create matplotlib plot with historical data and predictions"""
        plt.figure(figsize=(15, 8))
        
        # Plot historical data
        plt.plot(historical_df['date'], historical_df['close'], 
                label='Historical Price', linewidth=2, color='blue', alpha=0.7)
        
        # Plot predictions
        if not predictions_df.empty:
            plt.plot(predictions_df['date'], predictions_df['predicted_price'], 
                    label='Predicted Price', linewidth=2, color='red', linestyle='--')
            
            # Add confidence interval (simplified)
            confidence = predictions_df['predicted_price'] * 0.1  # 10% confidence interval
            plt.fill_between(predictions_df['date'], 
                           predictions_df['predicted_price'] - confidence,
                           predictions_df['predicted_price'] + confidence,
                           alpha=0.2, color='red', label='Confidence Interval')
        
        plt.title(f'{symbol} Price History and Predictions\n(Jan 2022 - Sep 2023 + 30-day Prediction)', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price (USD)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Convert plot to base64 string
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return plot_url

# Initialize predictor
predictor = CryptoPredictor()

# HTML template for the web page
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Crypto Price Predictions</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
        }
        .crypto-section {
            margin-bottom: 40px;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 8px;
        }
        .plot-container {
            text-align: center;
            margin: 20px 0;
        }
        .metrics {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .last-update {
            text-align: center;
            color: #666;
            font-style: italic;
            margin-top: 30px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Cryptocurrency Price Predictions</h1>
        <p style="text-align: center;">Historical data from January 2022 to September 2023 with 30-day predictions</p>
        
        {% for crypto in cryptos %}
        <div class="crypto-section">
            <h2>{{ crypto.symbol }}</h2>
            <div class="metrics">
                <strong>Model Performance:</strong><br>
                Mean Squared Error: {{ "%.2f"|format(crypto.mse) }}<br>
                R² Score: {{ "%.4f"|format(crypto.r2) }}<br>
                Data Points: {{ crypto.data_points }}
            </div>
            <div class="plot-container">
                <img src="data:image/png;base64,{{ crypto.plot_url }}" alt="{{ crypto.symbol }} Price Chart">
            </div>
        </div>
        {% endfor %}
        
        <div class="last-update">
            Last updated: {{ last_update }}
        </div>
    </div>
</body>
</html>
'''

def update_data():
    """Update cryptocurrency data and models periodically"""
    symbols = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT']
    
    while True:
        try:
            print("Updating cryptocurrency data...")
            
            for symbol in symbols:
                print(f"Fetching data for {symbol}...")
                
                # Fetch historical data
                df = predictor.fetch_crypto_data(symbol, '2022-01-01', '2023-09-30')
                
                if len(df) > 0:
                    # Prepare features
                    processed_df = predictor.prepare_features(df)
                    
                    # Train model
                    model, features, mse, r2 = predictor.train_model(symbol, processed_df)
                    
                    # Make future predictions
                    future_predictions = predictor.predict_future(model, features, processed_df, days_to_predict=30)
                    
                    # Create plot
                    plot_url = predictor.create_plot(symbol, processed_df, future_predictions)
                    
                    # Store results
                    predictor.crypto_data[symbol] = {
                        'historical_df': processed_df,
                        'predictions_df': future_predictions,
                        'model': model,
                        'mse': mse,
                        'r2': r2,
                        'plot_url': plot_url,
                        'data_points': len(processed_df)
                    }
                    
                    print(f"Successfully processed {symbol} with {len(processed_df)} data points")
                else:
                    print(f"No data fetched for {symbol}")
            
            print("Data update completed. Waiting 1 hour until next update...")
            time.sleep(3600)  # Update every hour
            
        except Exception as e:
            print(f"Error in update_data: {e}")
            time.sleep(300)  # Wait 5 minutes before retrying

@app.route('/')
def index():
    """Main page displaying crypto predictions"""
    cryptos = []
    
    for symbol in ['BTCUSDT', 'ETHUSDT', 'XRPUSDT']:
        if symbol in predictor.crypto_data:
            crypto_info = predictor.crypto_data[symbol]
            cryptos.append({
                'symbol': symbol.replace('USDT', ''),
                'plot_url': crypto_info['plot_url'],
                'mse': crypto_info['mse'],
                'r2': crypto_info['r2'],
                'data_points': crypto_info['data_points']
            })
    
    return render_template_string(HTML_TEMPLATE, 
                                cryptos=cryptos,
                                last_update=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == '__main__':
    # Start data update thread
    update_thread = threading.Thread(target=update_data, daemon=True)
    update_thread.start()
    
    # Wait a bit for initial data load
    print("Loading initial data...")
    time.sleep(10)
    
    # Start web server
    print("Starting web server on http://0.0.0.0:8080")
    app.run(host='0.0.0.0', port=8080, debug=False)
