from flask import Flask, render_template, jsonify
import requests
import os
from typing import List, Dict, Optional

app = Flask(__name__)

class FinnhubSymbolFetcher:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('FINNHUB_API_KEY')
        if not self.api_key:
            raise ValueError("Finnhub API key is required. Set FINNHUB_API_KEY environment variable.")
        
        self.base_url = "https://finnhub.io/api/v1"
    
    def get_stock_symbols(self, exchange: str = "US") -> List[Dict]:
        """Fetch stock symbols from Finnhub API for a specific exchange."""
        url = f"{self.base_url}/stock/symbol"
        params = {
            'exchange': exchange,
            'token': self.api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching symbols: {e}")
            return []
    
    def get_crypto_symbols(self) -> List[Dict]:
        """Fetch cryptocurrency symbols from Finnhub API."""
        url = f"{self.base_url}/crypto/symbol"
        params = {
            'exchange': 'binance',
            'token': self.api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching crypto symbols: {e}")
            return []

@app.route('/')
def index():
    """Main page displaying available symbols."""
    return render_template('index.html')

@app.route('/api/symbols/stocks')
def get_stock_symbols():
    """API endpoint to get stock symbols."""
    fetcher = FinnhubSymbolFetcher()
    symbols = fetcher.get_stock_symbols()
    return jsonify(symbols)

@app.route('/api/symbols/crypto')
def get_crypto_symbols():
    """API endpoint to get cryptocurrency symbols."""
    fetcher = FinnhubSymbolFetcher()
    symbols = fetcher.get_crypto_symbols()
    return jsonify(symbols)

@app.route('/api/symbols/all')
def get_all_symbols():
    """API endpoint to get both stock and crypto symbols."""
    fetcher = FinnhubSymbolFetcher()
    stocks = fetcher.get_stock_symbols()
    crypto = fetcher.get_crypto_symbols()
    
    return jsonify({
        'stocks': stocks,
        'crypto': crypto
    })

def create_template_directory():
    """Create templates directory if it doesn't exist."""
    os.makedirs('templates', exist_ok=True)

def create_index_template():
    """Create the HTML template for displaying symbols."""
    template_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Finnhub Symbols</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .tabs {
            display: flex;
            margin-bottom: 20px;
            border-bottom: 1px solid #ddd;
        }
        .tab {
            padding: 10px 20px;
            cursor: pointer;
            border: 1px solid transparent;
            border-bottom: none;
            margin-right: 5px;
            border-radius: 4px 4px 0 0;
        }
        .tab.active {
            background: #007bff;
            color: white;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        .symbol-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        .symbol-table th,
        .symbol-table td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        .symbol-table th {
            background-color: #f8f9fa;
            font-weight: bold;
        }
        .symbol-table tr:hover {
            background-color: #f8f9fa;
        }
        .loading {
            text-align: center;
            padding: 20px;
            color: #666;
        }
        .error {
            color: #dc3545;
            padding: 10px;
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            border-radius: 4px;
            margin: 10px 0;
        }
        .refresh-btn {
            background: #28a745;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            margin-bottom: 10px;
        }
        .refresh-btn:hover {
            background: #218838;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Finnhub Available Symbols</h1>
        
        <div class="tabs">
            <div class="tab active" onclick="showTab('stocks')">Stocks</div>
            <div class="tab" onclick="showTab('crypto')">Cryptocurrency</div>
            <div class="tab" onclick="showTab('all')">All Symbols</div>
        </div>

        <button class="refresh-btn" onclick="loadSymbols()">Refresh Symbols</button>

        <div id="stocks" class="tab-content active">
            <h2>Stock Symbols (US Exchange)</h2>
            <div id="stocks-content"></div>
        </div>

        <div id="crypto" class="tab-content">
            <h2>Cryptocurrency Symbols</h2>
            <div id="crypto-content"></div>
        </div>

        <div id="all" class="tab-content">
            <h2>All Symbols</h2>
            <div id="all-content"></div>
        </div>
    </div>

    <script>
        function showTab(tabName) {
            // Hide all tab contents
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Remove active class from all tabs
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Show selected tab content
            document.getElementById(tabName).classList.add('active');
            
            // Add active class to clicked tab
            event.target.classList.add('active');
            
            // Load data for the tab if not already loaded
            if (tabName === 'stocks' && !document.getElementById('stocks-content').innerHTML) {
                loadStocks();
            } else if (tabName === 'crypto' && !document.getElementById('crypto-content').innerHTML) {
                loadCrypto();
            } else if (tabName === 'all' && !document.getElementById('all-content').innerHTML) {
                loadAll();
            }
        }

        function loadSymbols() {
            loadStocks();
            loadCrypto();
            loadAll();
        }

        function loadStocks() {
            const content = document.getElementById('stocks-content');
            content.innerHTML = '<div class="loading">Loading stock symbols...</div>';
            
            fetch('/api/symbols/stocks')
                .then(response => response.json())
                .then(data => {
                    if (data.length === 0) {
                        content.innerHTML = '<div class="error">No stock symbols found or API error occurred.</div>';
                        return;
                    }
                    
                    let html = `<table class="symbol-table">
                        <thead>
                            <tr>
                                <th>Symbol</th>
                                <th>Description</th>
                                <th>Type</th>
                                <th>Currency</th>
                            </tr>
                        </thead>
                        <tbody>`;
                    
                    data.forEach(symbol => {
                        html += `<tr>
                            <td><strong>${symbol.symbol || 'N/A'}</strong></td>
                            <td>${symbol.description || 'N/A'}</td>
                            <td>${symbol.type || 'N/A'}</td>
                            <td>${symbol.currency || 'N/A'}</td>
                        </tr>`;
                    });
                    
                    html += '</tbody></table>';
                    content.innerHTML = html;
                })
                .catch(error => {
                    console.error('Error:', error);
                    content.innerHTML = `<div class="error">Error loading stock symbols: ${error.message}</div>`;
                });
        }

        function loadCrypto() {
            const content = document.getElementById('crypto-content');
            content.innerHTML = '<div class="loading">Loading cryptocurrency symbols...</div>';
            
            fetch('/api/symbols/crypto')
                .then(response => response.json())
                .then(data => {
                    if (data.length === 0) {
                        content.innerHTML = '<div class="error">No cryptocurrency symbols found or API error occurred.</div>';
                        return;
                    }
                    
                    let html = `<table class="symbol-table">
                        <thead>
                            <tr>
                                <th>Symbol</th>
                                <th>Description</th>
                                <th>Currency</th>
                            </tr>
                        </thead>
                        <tbody>`;
                    
                    data.forEach(symbol => {
                        html += `<tr>
                            <td><strong>${symbol.symbol || 'N/A'}</strong></td>
                            <td>${symbol.description || 'N/A'}</td>
                            <td>${symbol.currency || 'N/A'}</td>
                        </tr>`;
                    });
                    
                    html += '</tbody></table>';
                    content.innerHTML = html;
                })
                .catch(error => {
                    console.error('Error:', error);
                    content.innerHTML = `<div class="error">Error loading cryptocurrency symbols: ${error.message}</div>`;
                });
        }

        function loadAll() {
            const content = document.getElementById('all-content');
            content.innerHTML = '<div class="loading">Loading all symbols...</div>';
            
            fetch('/api/symbols/all')
                .then(response => response.json())
                .then(data => {
                    let html = '<h3>Stock Symbols</h3>';
                    
                    if (data.stocks && data.stocks.length > 0) {
                        html += `<table class="symbol-table">
                            <thead>
                                <tr>
                                    <th>Symbol</th>
                                    <th>Description</th>
                                    <th>Type</th>
                                    <th>Currency</th>
                                </tr>
                            </thead>
                            <tbody>`;
                        
                        data.stocks.forEach(symbol => {
                            html += `<tr>
                                <td><strong>${symbol.symbol || 'N/A'}</strong></td>
                                <td>${symbol.description || 'N/A'}</td>
                                <td>${symbol.type || 'N/A'}</td>
                                <td>${symbol.currency || 'N/A'}</td>
                            </tr>`;
                        });
                        
                        html += '</tbody></table>';
                    } else {
                        html += '<div class="error">No stock symbols found.</div>';
                    }
                    
                    html += '<h3>Cryptocurrency Symbols</h3>';
                    
                    if (data.crypto && data.crypto.length > 0) {
                        html += `<table class="symbol-table">
                            <thead>
                                <tr>
                                    <th>Symbol</th>
                                    <th>Description</th>
                                    <th>Currency</th>
                                </tr>
                            </thead>
                            <tbody>`;
                        
                        data.crypto.forEach(symbol => {
                            html += `<tr>
                                <td><strong>${symbol.symbol || 'N/A'}</strong></td>
                                <td>${symbol.description || 'N/A'}</td>
                                <td>${symbol.currency || 'N/A'}</td>
                            </tr>`;
                        });
                        
                        html += '</tbody></table>';
                    } else {
                        html += '<div class="error">No cryptocurrency symbols found.</div>';
                    }
                    
                    content.innerHTML = html;
                })
                .catch(error => {
                    console.error('Error:', error);
                    content.innerHTML = `<div class="error">Error loading symbols: ${error.message}</div>`;
                });
        }

        // Load initial data
        document.addEventListener('DOMContentLoaded', function() {
            loadStocks();
        });
    </script>
</body>
</html>'''
    
    with open('templates/index.html', 'w') as f:
        f.write(template_content)

if __name__ == '__main__':
    # Create template directory and file
    create_template_directory()
    create_index_template()
    
    # Check if API key is available
    api_key = os.getenv('FINNHUB_API_KEY')
    if not api_key:
        print("Warning: FINNHUB_API_KEY environment variable not set.")
        print("Please set your Finnhub API key before running:")
        print("export FINNHUB_API_KEY='your_api_key_here'")
        print("\nThe server will start, but API calls will fail without a valid API key.")
    
    # Start the web server
    print("Starting Finnhub Symbols Web Server...")
    print("Access the application at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
