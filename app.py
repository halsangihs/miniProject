# Flask Backend for Power Demand Prediction
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import pandas as pd
import holidays
from datetime import datetime, timedelta
import requests

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Load models and encoders at startup
print("Loading models...")
with open("final_rf_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("ordinal_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

print("✅ Models loaded successfully!")


def get_temperature_delhi(target_dt):
    """Fetch temperature for Delhi"""
    lat, lon = 28.7041, 77.1025
    now = datetime.now()
    yesterday = now - timedelta(days=1)
    
    # Historical data
    if target_dt < yesterday:
        try:
            date_str = target_dt.strftime("%Y-%m-%d")
            url = (
                "https://archive-api.open-meteo.com/v1/archive"
                f"?latitude={lat}&longitude={lon}"
                f"&start_date={date_str}&end_date={date_str}"
                f"&hourly=temperature_2m&timezone=Asia/Kolkata"
            )
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            times = data.get("hourly", {}).get("time", [])
            temps = data.get("hourly", {}).get("temperature_2m", [])
            
            if times and temps:
                target_hour_str = target_dt.strftime("%Y-%m-%dT%H")
                for i, time_str in enumerate(times):
                    if time_str.startswith(target_hour_str):
                        return temps[i]
                return round(sum(temps) / len(temps), 1)
        except:
            pass
    
    # Recent/Future forecast
    elif target_dt <= now + timedelta(days=7):
        try:
            date_str = target_dt.strftime("%Y-%m-%d")
            url = (
                "https://api.open-meteo.com/v1/forecast"
                f"?latitude={lat}&longitude={lon}"
                f"&hourly=temperature_2m"
                f"&start_date={date_str}&end_date={date_str}"
                "&timezone=Asia/Kolkata"
            )
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            times = data.get("hourly", {}).get("time", [])
            temps = data.get("hourly", {}).get("temperature_2m", [])
            
            if times and temps:
                target_hour_str = target_dt.strftime("%Y-%m-%dT%H")
                for i, time_str in enumerate(times):
                    if time_str.startswith(target_hour_str):
                        return temps[i]
                return round(sum(temps) / len(temps), 1)
        except:
            pass
    
    # Fallback to monthly averages
    month_avg = {1:14, 2:17, 3:22, 4:28, 5:32, 6:33, 7:31, 8:30, 9:30, 10:28, 11:22, 12:17}
    return month_avg.get(target_dt.month, 25)


def prepare_features(dt):
    """Prepare all features for prediction"""
    
    # Create dataframe
    df = pd.DataFrame([{'datetime': dt}])
    
    # Extract date features
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['date'] = df['datetime'].dt.day
    df['day'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = df['day'].isin([5, 6]).astype(int)
    
    # Check holidays
    indian_holidays = holidays.India(years=range(2021, 2035))
    df['is_holiday'] = df['datetime'].dt.normalize().isin(indian_holidays).astype(int)
    
    # Festival features
    dt_date = dt.date()
    if dt_date in indian_holidays:
        is_festival = 1
        festival_name = indian_holidays[dt_date]
        
        # Find same festival last year
        last_year = dt.year - 1
        festival_last_year_date = None
        for date, name in indian_holidays.items():
            if name == festival_name and date.year == last_year:
                festival_last_year_date = date
                break
        
        if festival_last_year_date:
            festival_day_last_year = festival_last_year_date.day
            festival_month_last_year = festival_last_year_date.month
        else:
            festival_day_last_year = -1
            festival_month_last_year = -1
    else:
        is_festival = 0
        festival_name = "No Festival"
        festival_day_last_year = -1
        festival_month_last_year = -1
    
    df['is_festival'] = is_festival
    df['festival_name'] = festival_name
    df['festival_day_last_year'] = festival_day_last_year
    df['festival_month_last_year'] = festival_month_last_year
    
    # Festival last year
    dt_last_year = dt - timedelta(days=365)
    dt_last_year_date = dt_last_year.date()
    df['is_festival_last_year'] = 1 if dt_last_year_date in indian_holidays else 0
    
    # Encode festival name
    current_festival = df['festival_name'].iloc[0]
    if hasattr(encoder, 'categories_'):
        known_festivals = list(encoder.categories_[0])
        if current_festival not in known_festivals:
            df['festival_name'] = 'No Festival'
    
    df['festival_name'] = encoder.transform(df[['festival_name']])[0]
    
    # Get temperature
    df['temp'] = get_temperature_delhi(dt)
    
    # Set datetime as index
    df = df.set_index('datetime')
    
    # Load historical data for lag features
    historical_df = pd.read_csv('./data/dehli_energy.csv', parse_dates=['datetime'], index_col='datetime')
    default_power_mean = float(historical_df['Power demand'].mean())
    
    dt_idx = df.index[0]
    
    # Lag features
    one_week_ago = dt_idx - pd.Timedelta(hours=168)
    try:
        df.loc[dt_idx, 'power_demand_1_week_ago'] = float(historical_df.loc[one_week_ago, 'Power demand'])
    except KeyError:
        df.loc[dt_idx, 'power_demand_1_week_ago'] = default_power_mean
    
    one_year_ago = dt_idx - pd.Timedelta(hours=8760)
    try:
        df.loc[dt_idx, 'power_demand_1_year_ago'] = float(historical_df.loc[one_year_ago, 'Power demand'])
    except KeyError:
        df.loc[dt_idx, 'power_demand_1_year_ago'] = default_power_mean
    
    df.loc[dt_idx, 'moving_avg_3h'] = float(historical_df['Power demand'].tail(3).mean())
    
    # Select features in correct order
    feature_cols = [
        'temp', 'year', 'month', 'date', 'day', 'is_weekend', 'is_holiday',
        'moving_avg_3h', 'power_demand_1_week_ago', 'power_demand_1_year_ago',
        'is_festival', 'festival_name', 'festival_month_last_year', 
        'festival_day_last_year', 'is_festival_last_year'
    ]
    
    return df[feature_cols]


@app.route('/')
def home():
    """Serve the frontend"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for predictions"""
    try:
        # Get input from request
        data = request.get_json()
        datetime_str = data.get('datetime')
        
        if not datetime_str:
            return jsonify({'error': 'datetime is required'}), 400
        
        # Parse datetime
        try:
            dt = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return jsonify({'error': 'Invalid datetime format. Use YYYY-MM-DD HH:MM:SS'}), 400
        
        # Prepare features
        features = prepare_features(dt)
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Prepare response
        response = {
            'success': True,
            'datetime': datetime_str,
            'predicted_power_demand': round(float(prediction), 2),
            'unit': 'MW',
            'features': {
                'temperature': round(float(features['temp'].iloc[0]), 2),
                'is_weekend': int(features['is_weekend'].iloc[0]),
                'is_holiday': int(features['is_holiday'].iloc[0]),
                'is_festival': int(features['is_festival'].iloc[0])
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': True})


if __name__ == '__main__':
    print("🚀 Starting Flask server...")
    print("📍 Server will run on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
