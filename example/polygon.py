import requests
import csv
import os
import time
from datetime import datetime, timedelta
from pypnf import PointFigureChart

class PolygonDataManager:
    def __init__(self, data_dir='market_data'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.base_url = "https://api.polygon.io"
        
        # Supported timeframes with their Polygon API multipliers
        self.timeframes = {
            '1min': {'multiplier': 1, 'timespan': 'minute', 'chart_title': '1 Minute'},
            '5min': {'multiplier': 5, 'timespan': 'minute', 'chart_title': '5 Minute'},
            '10min': {'multiplier': 10, 'timespan': 'minute', 'chart_title': '10 Minute'},
            '30min': {'multiplier': 30, 'timespan': 'minute', 'chart_title': '30 Minute'},
            '1hour': {'multiplier': 1, 'timespan': 'hour', 'chart_title': '1 Hour'},
            '2hour': {'multiplier': 2, 'timespan': 'hour', 'chart_title': '2 Hour'},
            '4hour': {'multiplier': 4, 'timespan': 'hour', 'chart_title': '4 Hour'},
            'daily': {'multiplier': 1, 'timespan': 'day', 'chart_title': 'Daily'}
        }
        
        # Supported scaling methods with descriptions
        self.scaling_methods = {
            'abs': {'name': 'Absolute', 'description': 'Fixed box size in price units'},
            'atr': {'name': 'ATR', 'description': 'Average True Range based scaling'},
            'cla': {'name': 'Classical', 'description': 'Classical variation scaling'},
            'log': {'name': 'Logarithmic', 'description': 'Logarithmic scaling for wide price ranges'},
        }
    
    def get_csv_filename(self, symbol, timeframe):
        """Generate CSV filename for symbol and timeframe"""
        clean_symbol = symbol.replace('^', '').replace('.', '_')
        return os.path.join(self.data_dir, f"{clean_symbol}_{timeframe}.csv")
    
    def load_existing_data(self, symbol, timeframe):
        """Load existing data from CSV for specific timeframe"""
        filename = self.get_csv_filename(symbol, timeframe)
        if not os.path.exists(filename):
            return None
        
        market_data = {
            'date': [], 
            'open': [], 
            'high': [], 
            'low': [], 
            'close': [], 
            'volume': []
        }
        
        try:
            with open(filename, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    market_data['date'].append(row['date'])
                    market_data['open'].append(float(row['open']))
                    market_data['high'].append(float(row['high']))
                    market_data['low'].append(float(row['low']))
                    market_data['close'].append(float(row['close']))
                    market_data['volume'].append(float(row['volume']))
            
            print(f"Loaded {len(market_data['date'])} existing {timeframe} records from {filename}")
            return market_data
        except Exception as e:
            print(f"Error loading existing {timeframe} data: {e}")
            return None
    
    def save_data_to_csv(self, symbol, timeframe, market_data):
        """Save complete dataset to CSV for specific timeframe"""
        filename = self.get_csv_filename(symbol, timeframe)
        
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['date', 'open', 'high', 'low', 'close', 'volume'])
            writer.writeheader()
            
            for i in range(len(market_data['date'])):
                writer.writerow({
                    'date': market_data['date'][i],
                    'open': market_data['open'][i],
                    'high': market_data['high'][i],
                    'low': market_data['low'][i],
                    'close': market_data['close'][i],
                    'volume': market_data['volume'][i]
                })
        
        print(f"Saved {len(market_data['date'])} {timeframe} records to {filename}")
    
    def get_polygon_symbol(self, symbol):
        return symbol
    
    def handle_api_error(self, response, attempt, max_retries):
        """Handle specific API error responses with detailed messages"""
        error_messages = {
            400: "Bad Request - Invalid parameters or malformed request",
            401: "Unauthorized - Invalid API key",
            402: "Payment Required - Subscription required for this endpoint",
            403: "Forbidden - API key doesn't have access to this endpoint",
            404: "Not Found - Symbol or endpoint not found",
            409: "Conflict - Data conflict error",
            429: "Too Many Requests - Rate limit exceeded",
            500: "Internal Server Error - Polygon.io server issue",
            502: "Bad Gateway - Upstream server error",
            503: "Service Unavailable - API temporarily unavailable",
            504: "Gateway Timeout - Upstream server timeout"
        }
        
        status_code = response.status_code
        error_msg = error_messages.get(status_code, f"HTTP {status_code} - Unknown error")
        
        # Try to get more details from response body
        try:
            error_data = response.json()
            print(f"🔍 RAW ERROR RESPONSE: {error_data}")
            
            # Extract error message from various possible fields
            error_fields = ['error', 'message', 'error_message', 'description', 'reason']
            for field in error_fields:
                if field in error_data:
                    error_msg += f": {error_data[field]}"
                    break
                    
            # If no specific error field found, show the entire response
            if error_msg == error_messages.get(status_code, f"HTTP {status_code} - Unknown error"):
                error_msg += f" - Full response: {error_data}"
                
        except Exception as e:
            print(f"Could not parse error response: {e}")
            error_msg += f" (Response body: {response.text[:500]})"
        
        print(f"API Error (Attempt {attempt + 1}/{max_retries}): {error_msg}")
        
        # Return whether this is a retryable error
        retryable_errors = [429, 500, 502, 503, 504]  # Rate limits and server errors
        return status_code in retryable_errors, error_msg
    
    def validate_api_response(self, data, endpoint):
        """Validate the structure and content of API responses"""
        if not isinstance(data, dict):
            raise Exception(f"Invalid API response: expected dict, got {type(data)}")
        
        if 'status' not in data:
            raise Exception("Invalid API response: missing 'status' field")
        
        # Accept both 'OK' and 'DELAYED' as successful statuses
        if data['status'] == 'ERROR':
            error_msg = data.get('error', data.get('error_message', data.get('message', 'Unknown API error')))
            raise Exception(f"Polygon API error: {error_msg}")
        
        # For delayed data, we still want to process it
        if data['status'] not in ['OK', 'DELAYED']:
            raise Exception(f"Unexpected API status: {data['status']}")
        
        # Validate specific endpoint responses
        if 'aggs' in endpoint:
            if 'results' not in data and data.get('resultsCount', 0) > 0:
                raise Exception("Invalid aggregates response: missing 'results' field")
            
            if 'results' in data and not isinstance(data['results'], list):
                raise Exception("Invalid aggregates response: 'results' is not a list")
        
        return True
    
    def fetch_with_backoff(self, url, params, max_retries=5, initial_delay=12):
        """
        Fetch data with comprehensive error handling and exponential backoff
        Polygon free tier: 5 calls/minute, so initial delay is 12 seconds
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                print(f"API Request (Attempt {attempt + 1}/{max_retries}): {url}")
                print(f"Request params: { {k: v for k, v in params.items() if k != 'apiKey'} }")
                
                response = requests.get(url, params=params, timeout=30)
                print(f"Response status code: {response.status_code}")
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        print(f"🔍 API Response Status: {data.get('status', 'N/A')}, Results Count: {data.get('resultsCount', 0)}")
                    except Exception as e:
                        raise Exception(f"Failed to parse JSON response: {e}")
                    
                    # Validate response structure
                    try:
                        self.validate_api_response(data, url)
                    except Exception as validation_error:
                        last_error = validation_error
                        print(f"Response validation failed: {validation_error}")
                        if attempt < max_retries - 1:
                            delay = initial_delay * (2 ** attempt)
                            print(f"Waiting {delay} seconds before retry...")
                            time.sleep(delay)
                            continue
                        else:
                            raise validation_error
                    
                    return data
                
                else:
                    # Handle API errors
                    is_retryable, error_msg = self.handle_api_error(response, attempt, max_retries)
                    last_error = Exception(error_msg)
                    
                    if is_retryable and attempt < max_retries - 1:
                        delay = initial_delay * (2 ** attempt)
                        print(f"Waiting {delay} seconds before retry...")
                        time.sleep(delay)
                        continue
                    else:
                        raise last_error
                    
            except requests.exceptions.Timeout:
                last_error = Exception(f"Request timeout (Attempt {attempt + 1}/{max_retries})")
                print(f"Request timeout, attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    delay = initial_delay * (2 ** attempt)
                    print(f"Waiting {delay} seconds before retry...")
                    time.sleep(delay)
                continue
                
            except requests.exceptions.ConnectionError:
                last_error = Exception(f"Connection error (Attempt {attempt + 1}/{max_retries})")
                print(f"Connection error, attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    delay = initial_delay * (2 ** attempt)
                    print(f"Waiting {delay} seconds before retry...")
                    time.sleep(delay)
                continue
                
            except requests.exceptions.RequestException as e:
                last_error = Exception(f"Request error: {e} (Attempt {attempt + 1}/{max_retries})")
                print(f"Request error: {e}, attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    delay = initial_delay * (2 ** attempt)
                    print(f"Waiting {delay} seconds before retry...")
                    time.sleep(delay)
                continue
                
            except Exception as e:
                last_error = e
                print(f"Unexpected error: {e}, attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    delay = initial_delay * (2 ** attempt)
                    print(f"Waiting {delay} seconds before retry...")
                    time.sleep(delay)
                continue
        
        # If we've exhausted all retries
        if last_error:
            raise Exception(f"Failed to fetch data after {max_retries} attempts. Last error: {last_error}")
        else:
            raise Exception(f"Failed to fetch data after {max_retries} attempts")
    
    def fetch_historical_batches(self, symbol, api_key, timeframe, months=6):
        """
        Fetch historical data in monthly batches for specific timeframe
        """
        polygon_symbol = self.get_polygon_symbol(symbol)
        tf_config = self.timeframes[timeframe]
        
        all_data = []
        current_date = datetime.now()
        
        print(f"Fetching {months} months of {timeframe} historical data for {polygon_symbol}...")
        
        for i in range(months):
            # Calculate date range for each batch (1 month each)
            end_date = current_date - timedelta(days=30 * i)
            start_date = end_date - timedelta(days=30)
            
            # Format dates for Polygon API (YYYY-MM-DD)
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            print(f"Fetching {timeframe} batch {i+1}/{months}: {start_str} to {end_str}")
            
            url = f"{self.base_url}/v2/aggs/ticker/{polygon_symbol}/range/{tf_config['multiplier']}/{tf_config['timespan']}/{start_str}/{end_str}"
            
            params = {
                'adjusted': 'true',
                'sort': 'asc',
                'limit': 50000,
                'apiKey': api_key
            }
            
            try:
                data = self.fetch_with_backoff(url, params, max_retries=3, initial_delay=12)
                
                # Accept both 'OK' and 'DELAYED' statuses
                if data['status'] in ['OK', 'DELAYED'] and data.get('resultsCount', 0) > 0:
                    batch_data = data['results']
                    all_data.extend(batch_data)
                    print(f"  {timeframe} Batch {i+1}: {len(batch_data)} records")
                else:
                    print(f"  {timeframe} Batch {i+1}: No data returned (status: {data.get('status')}, resultsCount: {data.get('resultsCount', 0)})")
                
            except Exception as e:
                print(f"  {timeframe} Batch {i+1}: Error - {e}")
                # Continue with next batch even if one fails
                continue
            
            # Respect Polygon free tier rate limits (5 calls per minute)
            if (i + 1) % 5 == 0 and (i + 1) < months:
                print("  Reached 5 API calls, waiting 60 seconds...")
                time.sleep(60)
            else:
                # Wait 12 seconds between calls to stay within 5 calls/minute
                time.sleep(12)
        
        if not all_data:
            raise Exception(f"No {timeframe} data received from any batch for {symbol}")
        
        # Convert to pypnf format
        market_data = {
            'date': [], 
            'open': [], 
            'high': [], 
            'low': [], 
            'close': [], 
            'volume': []
        }
        
        # Remove duplicates and sort by timestamp
        unique_data = {}
        for result in all_data:
            timestamp = result['t']
            if timestamp not in unique_data:
                unique_data[timestamp] = result
        
        sorted_timestamps = sorted(unique_data.keys())
        
        for timestamp in sorted_timestamps:
            result = unique_data[timestamp]
            dt = datetime.fromtimestamp(result['t'] / 1000)
            market_data['date'].append(dt.strftime('%Y-%m-%d %H:%M:%S'))
            market_data['open'].append(result['o'])
            market_data['high'].append(result['h'])
            market_data['low'].append(result['l'])
            market_data['close'].append(result['c'])
            market_data['volume'].append(result['v'])
        
        print(f"Total {timeframe} historical records: {len(market_data['date'])}")
        return market_data
    
    def fetch_latest_data(self, symbol, api_key, timeframe, days=7):
        """Fetch only the latest data for specific timeframe"""
        polygon_symbol = self.get_polygon_symbol(symbol)
        tf_config = self.timeframes[timeframe]
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        print(f"Fetching latest {days} days of {timeframe} data for {polygon_symbol}...")
        
        url = f"{self.base_url}/v2/aggs/ticker/{polygon_symbol}/range/{tf_config['multiplier']}/{tf_config['timespan']}/{start_str}/{end_str}"
        
        params = {
            'adjusted': 'true',
            'sort': 'asc',
            'limit': 50000,
            'apiKey': api_key
        }
        
        try:
            data = self.fetch_with_backoff(url, params, max_retries=3, initial_delay=12)
            
            # Accept both 'OK' and 'DELAYED' statuses
            if data['status'] not in ['OK', 'DELAYED']:
                error_msg = data.get('error', data.get('error_message', f"Unexpected status: {data['status']}"))
                raise Exception(f"Polygon API error: {error_msg}")
            
            if data.get('resultsCount', 0) == 0:
                print(f"No new {timeframe} data for {polygon_symbol} in the last {days} days")
                # Return empty data structure instead of raising exception
                return {
                    'date': [], 
                    'open': [], 
                    'high': [], 
                    'low': [], 
                    'close': [], 
                    'volume': []
                }
            
            new_data = {
                'date': [], 
                'open': [], 
                'high': [], 
                'low': [], 
                'close': [], 
                'volume': []
            }
            
            for result in data['results']:
                dt = datetime.fromtimestamp(result['t'] / 1000)
                new_data['date'].append(dt.strftime('%Y-%m-%d %H:%M:%S'))
                new_data['open'].append(result['o'])
                new_data['high'].append(result['h'])
                new_data['low'].append(result['l'])
                new_data['close'].append(result['c'])
                new_data['volume'].append(result['v'])
            
            print(f"Fetched {len(new_data['date'])} latest {timeframe} records from Polygon")
            return new_data
            
        except Exception as e:
            raise Exception(f"Failed to fetch latest {timeframe} data for {symbol}: {e}")
    
    def merge_data(self, existing_data, new_data):
        """Merge existing data with new data, removing duplicates"""
        if not existing_data:
            return new_data
        
        # Create a set of existing dates for fast lookup
        existing_dates = set(existing_data['date'])
        
        # Start with all existing data
        merged_data = {
            'date': existing_data['date'][:],
            'open': existing_data['open'][:],
            'high': existing_data['high'][:],
            'low': existing_data['low'][:],
            'close': existing_data['close'][:],
            'volume': existing_data['volume'][:]
        }
        
        # Add only new records that don't exist
        new_count = 0
        for i in range(len(new_data['date'])):
            if new_data['date'][i] not in existing_dates:
                merged_data['date'].append(new_data['date'][i])
                merged_data['open'].append(new_data['open'][i])
                merged_data['high'].append(new_data['high'][i])
                merged_data['low'].append(new_data['low'][i])
                merged_data['close'].append(new_data['close'][i])
                merged_data['volume'].append(new_data['volume'][i])
                new_count += 1
        
        # Sort by date to ensure chronological order
        if new_count > 0:
            sorted_indices = sorted(range(len(merged_data['date'])), 
                                  key=lambda i: merged_data['date'][i])
            
            sorted_data = {
                'date': [merged_data['date'][i] for i in sorted_indices],
                'open': [merged_data['open'][i] for i in sorted_indices],
                'high': [merged_data['high'][i] for i in sorted_indices],
                'low': [merged_data['low'][i] for i in sorted_indices],
                'close': [merged_data['close'][i] for i in sorted_indices],
                'volume': [merged_data['volume'][i] for i in sorted_indices]
            }
            
            print(f"Added {new_count} new records, total: {len(sorted_data['date'])}")
            return sorted_data
        else:
            print("No new records to add")
            return merged_data
    
    def get_data_with_update(self, symbol, api_key, timeframe='1hour', force_refresh=False, historical_months=6):
        """
        Main method: get data with smart updating for specific timeframe
        
        Parameters:
        symbol: Stock symbol
        api_key: Polygon.io API key
        timeframe: Timeframe to fetch ('1min', '5min', '10min', '30min', '1hour', '2hour', '4hour', 'daily')
        force_refresh: If True, fetch all data new
        historical_months: Number of months of historical data to fetch if no CSV exists
        
        Returns:
        dict: Complete market data for the specified timeframe
        """
        if timeframe not in self.timeframes:
            raise ValueError(f"Unsupported timeframe: {timeframe}. Supported: {list(self.timeframes.keys())}")
        
        if not api_key or api_key == "YOUR_POLYGON_API_KEY":
            raise ValueError("Valid Polygon.io API key required. Please provide your API key.")
        
        if force_refresh:
            print(f"Forcing full refresh with historical batches for {timeframe}...")
            try:
                historical_data = self.fetch_historical_batches(symbol, api_key, timeframe, historical_months)
                self.save_data_to_csv(symbol, timeframe, historical_data)
                return historical_data
            except Exception as e:
                raise Exception(f"Force refresh failed for {symbol} {timeframe}: {e}")
        
        # Load existing data
        existing_data = self.load_existing_data(symbol, timeframe)
        
        if not existing_data:
            # No existing data, fetch historical data in batches
            print(f"No existing {timeframe} data found, fetching historical data in batches...")
            try:
                historical_data = self.fetch_historical_batches(symbol, api_key, timeframe, historical_months)
                self.save_data_to_csv(symbol, timeframe, historical_data)
                return historical_data
            except Exception as e:
                raise Exception(f"Initial {timeframe} data fetch failed for {symbol}: {e}")
        
        # Fetch only latest data and merge
        print(f"Fetching latest {timeframe} data for update...")
        try:
            latest_data = self.fetch_latest_data(symbol, api_key, timeframe, days=7)
            merged_data = self.merge_data(existing_data, latest_data)
            
            # Save if we got new data
            if len(merged_data['date']) > len(existing_data['date']):
                self.save_data_to_csv(symbol, timeframe, merged_data)
            else:
                print(f"{timeframe} data is already up to date")
            
            return merged_data
            
        except Exception as e:
            print(f"Warning: Failed to fetch latest {timeframe} data, using existing data: {e}")
            return existing_data

    def get_recommended_boxsize(self, symbol, timeframe, scaling='abs'):
        """Get recommended boxsize based on symbol, timeframe and scaling method"""
        # Base boxsizes for different symbols (for absolute scaling)
        base_boxsizes = {
            '^GSPC': 2.0,   # S&P 500
            'AAPL': 0.5,     # Apple
            'MSFT': 1.0,     # Microsoft
            'TSLA': 0.5,     # Tesla
            'default': 1.0   # Default for other symbols
        }
        
        base_boxsize = base_boxsizes.get(symbol, base_boxsizes['default'])
        
        # Adjust for timeframe
        timeframe_multipliers = {
            '1min': 0.02, '5min': 0.05, '10min': 0.1, '30min': 0.25,
            '1hour': 0.5, '2hour': 0.75, '4hour': 1.0, 'daily': 2.0
        }
        
        boxsize = base_boxsize * timeframe_multipliers.get(timeframe, 1.0)
        
        # Adjust for scaling method
        scaling_multipliers = {
            'abs': 1.0,      # Absolute - no change
            'atr': 0.5,      # ATR based - smaller boxes
            'cla': 0.75,     # Classical - slightly smaller
            'log': 0.1,      # Logarithmic - much smaller
        }
        
        return boxsize * scaling_multipliers.get(scaling, 1.0)

    def create_chart(self, symbol, data, timeframe, scaling='abs', boxsize=None, reversal=3):
        """
        Create Point & Figure chart for specific timeframe with customizable scaling
        
        Parameters:
        symbol: Stock symbol
        data: Market data
        timeframe: Timeframe string
        scaling: Scaling method ('abs', 'atr', 'percent', 'log', 'volatility')
        boxsize: Manual boxsize (if None, calculated automatically)
        reversal: Number of boxes for reversal
        
        Returns:
        PointFigureChart object
        """
        if scaling not in self.scaling_methods:
            raise ValueError(f"Unsupported scaling: {scaling}. Supported: {list(self.scaling_methods.keys())}")
        
        # Calculate boxsize if not provided
        if boxsize is None:
            boxsize = self.get_recommended_boxsize(symbol, timeframe, scaling)
        
        tf_config = self.timeframes[timeframe]
        scaling_config = self.scaling_methods[scaling]
        box_unit = "%" if scaling == 'log' else "" if scaling == 'cla' else "pt"
        box_value = f"{scaling} {boxsize}{box_unit}" if scaling != 'cla' else ""
        title = f'{symbol} {timeframe} (polygon.io)'
        
        # Create chart with specified scaling
        pnf = PointFigureChart(
            ts=data, 
            method='h/l', 
            reversal=reversal, 
            boxsize=boxsize, 
            scaling=scaling, 
            title=title
        )
        
        # Add technical indicators
        pnf.bollinger(5, 2)
        pnf.donchian(8, 2)
        pnf.psar(0.02, 0.2)
        
        return pnf

    def create_multiple_scaling_charts(self, symbol, data, timeframe, scalings=None, boxsize=None, reversal=3):
        """
        Create multiple charts with different scaling methods
        
        Parameters:
        symbol: Stock symbol
        data: Market data
        timeframe: Timeframe string
        scalings: List of scaling methods to use
        boxsize: Manual boxsize (if None, calculated automatically)
        reversal: Number of boxes for reversal
        
        Returns:
        Dictionary of {scaling_method: chart_object}
        """
        if scalings is None:
            scalings = ['abs', 'percent', 'atr']
        
        charts = {}
        for scaling in scalings:
            try:
                chart = self.create_chart(symbol, data, timeframe, scaling, boxsize, reversal)
                charts[scaling] = chart
                print(f"✓ Created {timeframe} chart with {scaling} scaling")
            except Exception as e:
                print(f"✗ Failed to create {timeframe} chart with {scaling} scaling: {e}")
                charts[scaling] = None
        
        return charts

# Utility functions
def check_data_status(data_dir='market_data'):
    """Check what data we have and how recent it is for all timeframes"""
    data_manager = PolygonDataManager(data_dir)
    
    symbols = ['^GSPC', 'AAPL', 'MSFT', 'TSLA']
    timeframes = list(data_manager.timeframes.keys())
    
    for symbol in symbols:
        for timeframe in timeframes:
            filename = data_manager.get_csv_filename(symbol, timeframe)
            if os.path.exists(filename):
                data = data_manager.load_existing_data(symbol, timeframe)
                if data and data['date']:
                    latest_date = data['date'][-1]
                    print(f"{symbol} {timeframe}: {len(data['date'])} records, latest: {latest_date}")
                else:
                    print(f"{symbol} {timeframe}: No data or empty file")
            else:
                print(f"{symbol} {timeframe}: No data file found")

def update_with_multiple_scalings(symbol, api_key, timeframe='1hour', scalings=None, 
                                 force_refresh=False, historical_months=6):
    """Update data and create charts with multiple scaling methods"""
    data_manager = PolygonDataManager()
    
    if scalings is None:
        scalings = ['abs', 'percent', 'atr']
    
    print(f"\n{'='*60}")
    print(f"Processing {symbol} {timeframe} with scalings: {', '.join(scalings)}")
    print(f"{'='*60}")
    
    try:
        # Get data
        data = data_manager.get_data_with_update(
            symbol, 
            api_key, 
            timeframe=timeframe,
            force_refresh=force_refresh,
            historical_months=historical_months
        )
        
        # Create charts with different scalings
        charts = data_manager.create_multiple_scaling_charts(symbol, data, timeframe, scalings)
        
        results = {}
        for scaling, chart in charts.items():
            if chart is not None:
                chart_filename = f"chart_{symbol.replace('^', '')}_{timeframe}_{scaling}.html"
                chart.write_html(chart_filename)
                results[scaling] = {
                    'chart_file': chart_filename,
                    'records': len(data['date'])
                }
                print(f"✓ {scaling} scaling chart saved: {chart_filename}")
        
        return results
        
    except Exception as e:
        print(f"✗ Error processing {symbol} {timeframe}: {e}")
        return {'error': str(e)}

def update_all_timeframes_with_scalings(symbol, api_key, scalings=None, force_refresh=False, historical_months=6):
    """Update all timeframes for a symbol with multiple scaling methods"""
    data_manager = PolygonDataManager()
    
    if scalings is None:
        scalings = ['abs', 'percent']
    
    results = {}
    for timeframe in data_manager.timeframes.keys():
        try:
            print(f"\n{'='*60}")
            print(f"Processing {symbol} {timeframe}")
            print(f"{'='*60}")
            
            # Get data
            data = data_manager.get_data_with_update(
                symbol, 
                api_key, 
                timeframe=timeframe,
                force_refresh=force_refresh,
                historical_months=historical_months
            )
            
            # Create charts with different scalings
            timeframe_results = {}
            for scaling in scalings:
                try:
                    pnf = data_manager.create_chart(symbol, data, timeframe, scaling=scaling)
                    chart_filename = f"chart_{symbol.replace('^', '')}_{timeframe}_{scaling}.html"
                    pnf.write_html(chart_filename)
                    
                    timeframe_results[scaling] = {
                        'chart_file': chart_filename,
                        'records': len(data['date'])
                    }
                    
                    print(f"✓ {timeframe} {scaling}: {len(data['date'])} records -> {chart_filename}")
                    
                except Exception as e:
                    print(f"✗ {timeframe} {scaling}: {e}")
                    timeframe_results[scaling] = {'error': str(e)}
            
            results[timeframe] = timeframe_results
            
        except Exception as e:
            print(f"✗ Error processing {symbol} {timeframe}: {e}")
            results[timeframe] = {'error': str(e)}
    
    return results

def update_multiple_symbols_scalings(symbols, api_key, timeframes=None, scalings=None, 
                                   force_refresh=False, historical_months=6):
    """Update multiple symbols, timeframes, and scaling methods"""
    data_manager = PolygonDataManager()
    
    if timeframes is None:
        timeframes = ['1hour', '4hour', 'daily']
    if scalings is None:
        scalings = ['abs', 'percent']
    
    all_results = {}
    for symbol in symbols:
        print(f"\n{'#'*80}")
        print(f"PROCESSING SYMBOL: {symbol}")
        print(f"{'#'*80}")
        
        symbol_results = {}
        for timeframe in timeframes:
            timeframe_results = {}
            for scaling in scalings:
                try:
                    data = data_manager.get_data_with_update(
                        symbol, 
                        api_key, 
                        timeframe=timeframe,
                        force_refresh=force_refresh,
                        historical_months=historical_months
                    )
                    
                    # Create chart with specific scaling
                    pnf = data_manager.create_chart(symbol, data, timeframe, scaling=scaling)
                    chart_filename = f"chart_{symbol.replace('^', '')}_{timeframe}_{scaling}.html"
                    pnf.write_html(chart_filename)
                    
                    timeframe_results[scaling] = {
                        'records': len(data['date']),
                        'chart_file': chart_filename,
                        'date_range': f"{data['date'][0]} to {data['date'][-1]}" if data['date'] else 'No data'
                    }
                    
                    print(f"✓ {timeframe} {scaling}: {len(data['date'])} records -> {chart_filename}")
                    
                except Exception as e:
                    print(f"✗ {timeframe} {scaling}: {e}")
                    timeframe_results[scaling] = {'error': str(e)}
            
            symbol_results[timeframe] = timeframe_results
        
        all_results[symbol] = symbol_results
    
    return all_results

# Quick usage examples
def update_sp500_comprehensive(API_KEY=None):
    """Quick function to update S&P 500 with multiple timeframes and scalings"""
    if not API_KEY:
        raise ValueError("API_KEY is required")

    symbols = ['^GSPC']
    timeframes = ['1hour', '4hour', 'daily']
    scalings = ['abs', 'percent', 'atr']
    
    print("Updating S&P 500 with comprehensive analysis...")
    results = update_multiple_symbols_scalings(
        symbols, 
        API_KEY, 
        timeframes=timeframes,
        scalings=scalings,
        force_refresh=False,
        historical_months=3
    )
    
    return results

def create_custom_chart(symbol, timeframe, scaling='abs', boxsize=None, reversal=3, API_KEY=None, force_refresh=False, historical_months=6):
    """Create a single custom chart with specific parameters"""
    data_manager = PolygonDataManager()
    if not API_KEY:
        raise ValueError("API_KEY is required")

    try:
        data = data_manager.get_data_with_update(symbol, API_KEY, timeframe=timeframe, force_refresh=force_refresh, historical_months=historical_months)
        pnf = data_manager.create_chart(symbol, data, timeframe, scaling=scaling, boxsize=boxsize, reversal=reversal)
        
        scaling_suffix = f"_{scaling}"
        chart_filename = f"chart_{symbol.replace('^', '')}_{timeframe}{scaling_suffix}_{boxsize}_x_{reversal}.html"
        pnf.write_html(chart_filename)
        
        print(f"Custom chart saved: {chart_filename}")
        print(f"Parameters: timeframe={timeframe}, scaling={scaling}, boxsize={boxsize}, reversal={reversal}")
        return pnf
        
    except Exception as e:
        print(f"Error creating custom chart: {e}")
        return None

API_KEY = "YOUR_POLYGON_API_KEY"
pnf = create_custom_chart("SPY", "10min", scaling="log", boxsize=1, reversal=3, API_KEY=API_KEY, historical_months=24)
pnf.show()