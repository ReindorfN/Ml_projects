# Movie Recommendation App

A Flask-based movie recommendation system that uses machine learning to predict movie ratings and provide personalized recommendations.

## Features

- **Top-N Recommendations**: Get the highest-rated movie predictions based on ML model
- **Similar Movies**: Find movies similar to a selected movie using KNN similarity
- **Genre-based Recommendations**: Filter top recommendations by genre
- **Insight Dashboard**: View model performance metrics and data visualizations

## Deployment on Render

### Prerequisites

1. All required files are in place:
   - `app.py` - Main Flask application
   - `requirements.txt` - Python dependencies
   - `Procfile` - Tells Render how to run the app
   - `runtime.txt` - Python version specification (Python 3.11)
   - `render.yaml` - Render configuration (optional, but recommended)
   - `templates/` - HTML templates directory
   - `model_artifacts/` - Contains trained models and data files

### Deployment Steps

1. **Push to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Create Render Web Service**:
   - Go to [Render Dashboard](https://dashboard.render.com)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Configure:
     - **Name**: movie-recommendation-app (or your preferred name)
     - **Environment**: Python 3
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `gunicorn app:app` (or leave blank, Procfile handles this)
     - **Plan**: Free tier is fine for testing

3. **Environment Variables** (Optional):
   - `FLASK_ENV`: Set to `production` (or leave unset for production mode)
   - `PORT`: Automatically set by Render (don't override)
   - `PYTHON_VERSION`: Should be set to `3.11.0` via render.yaml

4. **Python Version**:
   - The app requires Python 3.11 (pandas 2.1.1 compatibility)
   - This is specified in `runtime.txt` and `render.yaml`
   - If Render still uses Python 3.13, you may need to:
     - Manually set Python version in Render dashboard: Settings → Environment → Python Version → 3.11.0
     - Or upgrade pandas in requirements.txt to version 2.2.0+ (supports Python 3.13)

4. **Deploy**:
   - Click "Create Web Service"
   - Render will automatically build and deploy your app
   - First deployment may take 5-10 minutes

### Important Notes

- **Model Artifacts**: Make sure all `.pkl` files and images in `model_artifacts/` are committed to your repository
- **File Size**: If model files are very large (>100MB), consider using Render's disk storage or external storage
- **Memory**: The app loads models into memory on startup. Free tier has 512MB RAM which should be sufficient for most models
- **Startup Time**: First request may be slow as models are loaded (lazy loading)
- **Python Version Compatibility**: If you encounter pandas compilation errors, ensure Python 3.11 is used (not 3.13)

### Troubleshooting

**Error: pandas compilation fails with Python 3.13**
- Solution 1: In Render dashboard, go to Settings → Environment → Python Version → Select `3.11.0`
- Solution 2: Upgrade pandas in requirements.txt:
  ```
  pandas>=2.2.0
  numpy>=1.26.0
  ```
  Then update runtime.txt to `python-3.13` if you want to use Python 3.13

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py

# Or with gunicorn (production-like)
gunicorn app:app
```

## Project Structure

```
Ml_projects/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── Procfile              # Render deployment config
├── runtime.txt           # Python version
├── .gitignore           # Git ignore rules
├── README.md            # This file
├── templates/           # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── similar.html
│   ├── by_genre.html
│   └── dashboard.html
└── model_artifacts/     # ML models and data
    ├── best_model.pkl
    ├── scaler.pkl
    ├── movie_data.pkl
    ├── movie_titles.pkl
    └── *.png            # Visualization images
```

## Technologies

- **Flask**: Web framework
- **scikit-learn**: Machine learning models
- **pandas & numpy**: Data processing
- **gunicorn**: WSGI HTTP server for production
