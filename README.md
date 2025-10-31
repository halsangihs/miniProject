# Delhi Power Demand Predictor

A machine learning web application that predicts power demand for Delhi based on date, time, weather, and festival information.

## 🚀 Features

- Real-time power demand prediction
- Beautiful animated UI with grid pattern and floating bubbles
- Weather data integration using Open-Meteo API
- Festival and holiday detection
- Responsive design for all devices

## 📋 Prerequisites

- Python 3.8 or higher
- Git

## 🛠️ Local Setup

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and visit:
```
http://localhost:5000
```

## 📦 Project Structure

```
miniProjecct/
├── app.py                      # Flask backend application
├── templates/
│   └── index.html             # Frontend UI
├── data/
│   ├── dehli_energy.csv       # Historical energy data
│   └── hourlyData.csv         # Hourly consumption data
├── final_rf_model.pkl         # Trained Random Forest model
├── ordinal_encoder.pkl        # Festival name encoder
├── main.ipynb                 # Training notebook
├── second.ipynb               # Pipeline development notebook
├── output.ipynb               # Inference testing notebook
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🌐 Deployment Options

### Option 1: Deploy to Render (Recommended - Free)

1. Create account at [render.com](https://render.com)
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Configure:
   - **Name**: delhi-power-predictor
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
5. Add `gunicorn` to requirements.txt first:
   ```bash
   echo "gunicorn==21.2.0" >> requirements.txt
   ```
6. Click "Create Web Service"

### Option 2: Deploy to Railway

1. Create account at [railway.app](https://railway.app)
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your repository
4. Railway auto-detects Python and deploys
5. Your app will be live in minutes!

### Option 3: Deploy to Heroku

1. Install Heroku CLI
2. Create `Procfile`:
   ```
   web: gunicorn app:app
   ```
3. Deploy:
   ```bash
   heroku login
   heroku create your-app-name
   git push heroku main
   ```

### Option 4: Deploy to PythonAnywhere

1. Create account at [pythonanywhere.com](https://www.pythonanywhere.com)
2. Upload your files
3. Set up web app with Flask
4. Configure WSGI file to point to `app.py`

## 🔧 Environment Variables

No environment variables required - the app works out of the box!

## 📊 Model Information

- **Algorithm**: Random Forest Regressor
- **Features**: 15 features including date, time, temperature, holidays, festivals, and lag features
- **Dataset**: Delhi energy consumption historical data

## 🎨 Technologies Used

- **Backend**: Flask, Python
- **Frontend**: HTML5, CSS3, JavaScript
- **ML Libraries**: scikit-learn, pandas, numpy
- **APIs**: Open-Meteo Weather API

## 📝 API Endpoints

- `GET /` - Serves the web interface
- `POST /predict` - Accepts datetime and returns power demand prediction
- `GET /health` - Health check endpoint

## 🤝 Contributing

Feel free to fork this project and submit pull requests!

## 📄 License

MIT License

## 👤 Author

Your Name - [GitHub Profile](https://github.com/YOUR_USERNAME)

## 🐛 Issues

Found a bug? Please open an issue on GitHub!
