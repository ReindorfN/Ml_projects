import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# --- Configuration and Initialization ---
app = Flask(__name__)
MODEL_DIR = 'model_artifacts'
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..') if os.path.dirname(__file__) else '..'

# Global variables to store loaded assets
MODEL = None
SCALER = None
MOVIE_DATA = None
MOVIE_TITLES = []
KNN_MODEL = None
FEATURE_COLUMNS = None

def load_assets():
    """Load the trained model, scaler, and movie data."""
    global MODEL, SCALER, MOVIE_DATA, MOVIE_TITLES, KNN_MODEL, FEATURE_COLUMNS
    
    try:
        # Try to load from model_artifacts first
        model_path = os.path.join(MODEL_DIR, 'best_model.pkl')
        scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
        data_path = os.path.join(MODEL_DIR, 'movie_data.pkl')
        titles_path = os.path.join(MODEL_DIR, 'movie_titles.pkl')
        
        if os.path.exists(model_path):
            # 1. Load the best regression model (e.g., Random Forest)
            with open(model_path, 'rb') as f:
                MODEL = pickle.load(f)
            print("✅ Model loaded from model_artifacts")
        else:
            print("⚠️  Model file not found. Will use fallback data loading.")
            
        if os.path.exists(scaler_path):
            # 2. Load the fitted scaler (crucial for Ridge/KNN)
            with open(scaler_path, 'rb') as f:
                SCALER = pickle.load(f)
            print("✅ Scaler loaded from model_artifacts")
        else:
            SCALER = StandardScaler()
            print("⚠️  Scaler not found. Using new StandardScaler instance.")
            
        if os.path.exists(data_path):
            # 3. Load the pre-processed movie feature matrix (X) and titles
            with open(data_path, 'rb') as f:
                MOVIE_DATA = pickle.load(f)
            print("✅ Movie data loaded from model_artifacts")
        else:
            print("⚠️  Pre-processed data not found. Loading from raw CSV files...")
            MOVIE_DATA = load_and_prepare_data_from_csv()
            print("✅ Movie data prepared from raw CSV files")
        
        if os.path.exists(titles_path):
            #loading the movie titles
            with open(titles_path, 'rb') as f:
                MOVIE_TITLES = pickle.load(f)
            print("✅ Movie titles loaded from model_artifacts")
        else:
            print("⚠️  Movie titles not found. Loading from raw CSV files...")
            MOVIE_TITLES = load_and_prepare_data_from_csv()
            print("✅ Movie titles prepared from raw CSV files")

        # Prepare features for similarity search
        if MOVIE_DATA is not None:
            prepare_similarity_model()
        
        print("✅ ML assets loaded successfully.")
        
    except Exception as e:
        print(f"❌ Error loading assets: {e}")
        print("Attempting to load from raw CSV files as fallback...")
        try:
            MOVIE_DATA = load_and_prepare_data_from_csv()
            MOVIE_TITLES = MOVIE_DATA['title'].tolist() if MOVIE_DATA is not None else []
            prepare_similarity_model()
            print("✅ Fallback data loading successful.")
        except Exception as e2:
            print(f"❌ Fallback loading also failed: {e2}")
            MOVIE_DATA = None
            MOVIE_TITLES = []
        
# --- Helper Functions ---
def load_and_prepare_data_from_csv():
    """Load movies and ratings from CSV files and prepare features."""
    try:
        # Determine the correct path to CSV files
        base_path = os.path.dirname(os.path.dirname(__file__))
        movies_path = os.path.join(base_path, 'movies.csv')
        ratings_path = os.path.join(base_path, 'ratings.csv')
        
        if not os.path.exists(movies_path):
            # Try alternative path
            movies_path = os.path.join(DATA_DIR, 'movies.csv')
            ratings_path = os.path.join(DATA_DIR, 'ratings.csv')
        
        if not os.path.exists(movies_path):
            print(f"❌ Could not find movies.csv at {movies_path}")
            return None
        
        # Load movies
        movies_df = pd.read_csv(movies_path)
        
        # Load ratings and calculate average ratings
        if os.path.exists(ratings_path):
            ratings_df = pd.read_csv(ratings_path)
            movie_stats = ratings_df.groupby('movieId').agg(
                avg_rating=('rating', 'mean'),
                rating_count=('rating', 'count')
            ).reset_index()
            
            # Filter movies with at least 50 ratings
            MIN_RATINGS = 50
            qualified = movie_stats[movie_stats['rating_count'] >= MIN_RATINGS]
            movies_df = pd.merge(movies_df, qualified, on='movieId', how='inner')
        else:
            # If no ratings file, create dummy stats
            movies_df['avg_rating'] = 3.5
            movies_df['rating_count'] = 100
            print("⚠️  ratings.csv not found. Using placeholder ratings.")
        
        # Create genre one-hot encoding
        movie_genres_ohe = movies_df['genres'].str.get_dummies(sep='|')
        if '(no genres listed)' in movie_genres_ohe.columns:
            movie_genres_ohe = movie_genres_ohe.drop(columns=['(no genres listed)'])
        
        # Add log rating count
        movies_df['log_rating_count'] = np.log1p(movies_df['rating_count'])
        
        # Combine all features
        feature_cols = list(movie_genres_ohe.columns) + ['log_rating_count']
        movies_df = pd.concat([movies_df, movie_genres_ohe], axis=1)
        
        return movies_df
        
    except Exception as e:
        print(f"❌ Error loading data from CSV: {e}")
        return None

def prepare_similarity_model():
    """Prepare KNN model for similarity search."""
    global KNN_MODEL, FEATURE_COLUMNS
    
    if MOVIE_DATA is None:
        return
    
    try:
        # Identify feature columns in the CORRECT ORDER to match training
        # Training order: genre OHE columns + log_rating_count (log_rating_count is LAST)
        metadata_cols = ['movieId', 'title', 'genres', 'avg_rating', 'rating_count']
        
        # Get genre columns (all columns that are not metadata and not log_rating_count)
        genre_cols = [col for col in MOVIE_DATA.columns 
                     if col not in metadata_cols and col != 'log_rating_count']
        
        # Ensure log_rating_count exists
        if 'log_rating_count' not in MOVIE_DATA.columns:
            print("⚠️  Warning: log_rating_count not found. Recalculating...")
            MOVIE_DATA['log_rating_count'] = np.log1p(MOVIE_DATA['rating_count'])
        
        # Feature columns in correct order: genre columns FIRST, then log_rating_count LAST
        FEATURE_COLUMNS = genre_cols + ['log_rating_count']
        
        if not FEATURE_COLUMNS:
            print("⚠️  No feature columns found for similarity search.")
            return
        
        # Extract features in the correct order
        X_features = MOVIE_DATA[FEATURE_COLUMNS].values
        
        # Scale features
        if SCALER and hasattr(SCALER, 'mean_'):
            # Use existing scaler if fitted
            X_scaled = SCALER.transform(X_features)
        else:
            # Fit new scaler
            SCALER.fit(X_features)
            X_scaled = SCALER.transform(X_features)
        
        # Train KNN model for similarity search
        KNN_MODEL = NearestNeighbors(n_neighbors=11, metric='cosine', algorithm='brute')
        KNN_MODEL.fit(X_scaled)
        print("✅ Similarity model (KNN) prepared successfully.")
        
    except Exception as e:
        print(f"⚠️  Error preparing similarity model: {e}")

def get_top_n_recommendations(n=10):
    """Predicts ratings for all movies and returns the top N highest-rated."""
    if MOVIE_DATA is None:
        return []

    # If model is available, use it for predictions
    if MODEL is not None:
        try:
            # Prepare features in the CORRECT ORDER to match training
            # Training order: genre OHE columns + log_rating_count (log_rating_count is LAST)
            # Exclude metadata columns
            metadata_cols = ['movieId', 'title', 'genres', 'avg_rating', 'rating_count']
            
            # Get genre columns (all columns that are not metadata and not log_rating_count)
            genre_cols = [col for col in MOVIE_DATA.columns 
                         if col not in metadata_cols and col != 'log_rating_count']
            
            # Ensure log_rating_count exists
            if 'log_rating_count' not in MOVIE_DATA.columns:
                print("⚠️  Warning: log_rating_count not found. Recalculating...")
                MOVIE_DATA['log_rating_count'] = np.log1p(MOVIE_DATA['rating_count'])
            
            # Feature columns in correct order: genre columns FIRST, then log_rating_count LAST
            # This matches training: X = pd.concat([movie_genres_ohe, modeling_df[['log_rating_count']]], axis=1)
            feature_cols = genre_cols + ['log_rating_count']
            
            if not feature_cols:
                # Fallback to using avg_rating if no features
                results = MOVIE_DATA[['movieId', 'title', 'genres', 'avg_rating']].copy()
                results['predicted_rating'] = results['avg_rating']
            else:
                # Extract features in the correct order
                X_predict = MOVIE_DATA[feature_cols]
                
                # Check if model is RandomForest (doesn't need scaling) or Ridge/KNN (needs scaling)
                # RandomForest was trained on unscaled data, so we don't scale for it
                model_type = type(MODEL).__name__
                if model_type == 'RandomForestRegressor':
                    # RandomForest doesn't need scaling - use features as-is
                    predicted_ratings = MODEL.predict(X_predict)
                else:
                    # Ridge/KNN need scaling
                    if SCALER and hasattr(SCALER, 'mean_'):
                        X_predict_scaled = SCALER.transform(X_predict)
                        predicted_ratings = MODEL.predict(X_predict_scaled)
                    else:
                        # If scaler not available, try without scaling
                        predicted_ratings = MODEL.predict(X_predict)
                
                # Combine predictions with metadata
                results = MOVIE_DATA[['movieId', 'title', 'genres']].copy()
                results['predicted_rating'] = predicted_ratings
        except Exception as e:
            print(f"⚠️  Error in model prediction: {e}. Using average ratings instead.")
            results = MOVIE_DATA[['movieId', 'title', 'genres', 'avg_rating']].copy()
            results['predicted_rating'] = results['avg_rating']
    else:
        # Fallback: Use average ratings from data
        if 'avg_rating' in MOVIE_DATA.columns:
            results = MOVIE_DATA[['movieId', 'title', 'genres', 'avg_rating']].copy()
            results['predicted_rating'] = results['avg_rating']
        else:
            return []
    
    # Sort and return top N
    top_n = results.sort_values(by='predicted_rating', ascending=False).head(n)
    
    # Convert to a list of dicts for easy template rendering
    return top_n.to_dict('records')


def get_similar_movies(movie_id, n=10):
    """Finds N nearest neighbors to a given movie in the feature space."""
    if MOVIE_DATA is None or KNN_MODEL is None or FEATURE_COLUMNS is None:
        return []
    
    if movie_id not in MOVIE_DATA['movieId'].values:
        return []

    try:
        # Get the source movie's features
        movie_idx = MOVIE_DATA[MOVIE_DATA['movieId'] == movie_id].index[0]
        source_features = MOVIE_DATA.loc[movie_idx, FEATURE_COLUMNS].values.reshape(1, -1)
        
        # Scale features
        if SCALER and hasattr(SCALER, 'mean_'):
            source_features = SCALER.transform(source_features)
        
        # Find nearest neighbors (n+1 because the movie itself will be included)
        distances, indices = KNN_MODEL.kneighbors(source_features, n_neighbors=min(n+1, len(MOVIE_DATA)))
        
        # Get similar movies (exclude the movie itself)
        similar_movies = []
        for i, idx in enumerate(indices[0][1:], 1):  # Skip first (the movie itself)
            if i > n:
                break
            movie_row = MOVIE_DATA.iloc[idx]
            # Convert cosine distance to similarity score (1 - distance)
            similarity = 1 - distances[0][i] if distances[0][i] <= 1 else 0
            similar_movies.append({
                'movieId': int(movie_row['movieId']),
                'title': movie_row['title'],
                'genres': movie_row.get('genres', 'N/A'),
                'similarity_score': max(0, min(1, similarity))  # Clamp between 0 and 1
            })
        
        return similar_movies
        
    except Exception as e:
        print(f"⚠️  Error finding similar movies: {e}")
        return []

def get_available_genres():
    """Extract all available genres from MOVIE_DATA."""
    if MOVIE_DATA is None or 'genres' not in MOVIE_DATA.columns:
        return []
    
    try:
        # Get all unique genres by splitting pipe-separated strings
        all_genres = set()
        for genres_str in MOVIE_DATA['genres'].dropna():
            if genres_str and genres_str != '(no genres listed)':
                genres_list = genres_str.split('|')
                all_genres.update(genres_list)
        
        # Remove the special case
        all_genres.discard('(no genres listed)')
        
        # Return sorted list
        return sorted(list(all_genres))
    except Exception as e:
        print(f"⚠️  Error extracting genres: {e}")
        return []

def get_top_n_by_genre(genre, n=10):
    """Get top N movie recommendations filtered by genre."""
    if MOVIE_DATA is None or not genre:
        return []
    
    # Get a larger set of top recommendations to ensure we have enough after filtering
    # We'll get more than n to account for filtering
    top_recommendations = get_top_n_recommendations(n=1000)  # Get top 100 to filter from
    
    if not top_recommendations:
        return []
    
    # Filter by genre: check if the genre string contains the selected genre
    filtered = []
    for movie in top_recommendations:
        movie_genres = movie.get('genres', '')
        if movie_genres and genre in movie_genres.split('|'):
            filtered.append(movie)
            if len(filtered) >= n:  # Stop once we have enough
                break
    
    return filtered

# --- Flask Routes ---

_assets_loaded = False

def ensure_assets_loaded():
    """Ensure assets are loaded (lazy loading, but only once)."""
    global _assets_loaded
    if not _assets_loaded:
        load_assets()
        _assets_loaded = True

@app.route('/')
def index():
    """Home page: Displays Top-N Predicted Recommendations."""
    ensure_assets_loaded()
    top_recommendations = get_top_n_recommendations(n=15)
    
    return render_template(
        'index.html', 
        page_title="Home | Top Recommendations", 
        recommendations=top_recommendations
    )

@app.route('/similar', methods=['GET', 'POST'])
def similar():
    """Similar Movies Page: Allows searching for movie neighbors."""
    ensure_assets_loaded()
    similar_movies = []
    selected_movie = None
    selected_movie_id = None
    
    if request.method == 'POST':
        selected_title = request.form.get('movie_search', '').strip()
        
        if selected_title and MOVIE_DATA is not None:
            # Look up the movieId from the title
            movie_match = MOVIE_DATA[MOVIE_DATA['title'] == selected_title]
            
            if not movie_match.empty:
                movie_id = movie_match['movieId'].iloc[0]
                selected_movie = selected_title
                selected_movie_id = int(movie_id)
                similar_movies = get_similar_movies(movie_id, n=10)

    return render_template(
        'similar.html', 
        page_title="Find Similar Movies",
        movie_titles=MOVIE_TITLES, # For dropdown/autocomplete
        similar_movies=similar_movies,
        selected_movie=selected_movie,
        selected_movie_id=selected_movie_id
    )

@app.route('/api/search_movies')
def api_search_movies():
    """API endpoint for movie search autocomplete."""
    ensure_assets_loaded()
    query = request.args.get('q', '').strip().lower()
    
    if not query or not MOVIE_TITLES:
        return jsonify([])
    
    # Filter movies that contain the query
    matches = [title for title in MOVIE_TITLES if query in title.lower()][:20]
    return jsonify(matches)

@app.route('/model_artifacts/<path:filename>')
def serve_model_artifact(filename):
    """Serve images and other files from model_artifacts directory."""
    return send_from_directory(MODEL_DIR, filename)


@app.route('/by_genre', methods=['GET', 'POST'])
def by_genre():
    """By Genre Page: Displays top-N recommendations by genre."""
    ensure_assets_loaded()
    
    # Get available genres for the dropdown
    available_genres = get_available_genres()
    
    # Get selected genre from form or query parameter
    selected_genre = None
    recommendations = []
    
    if request.method == 'POST':
        selected_genre = request.form.get('genre', '').strip()
    else:
        selected_genre = request.args.get('genre', '').strip()
    
    # If a genre is selected, get filtered recommendations
    if selected_genre:
        recommendations = get_top_n_by_genre(selected_genre, n=10)
    
    return render_template(
        'by_genre.html',
        page_title="Top Recommendations by Genre",
        available_genres=available_genres,
        selected_genre=selected_genre,
        recommendations=recommendations
    )

@app.route('/dashboard')
def dashboard():
    """Insight Dashboard: Displays data analysis and model comparison charts."""
    ensure_assets_loaded()
    

    
    model_comparison_data = {
        'Model': ['RandomForest', 'Ridge', 'KNN'],
        'RMSE': [0.396510, 0.403917 , 0.411830],
        'MAE': [0.307487, 0.313320, 0.320998]
    }
    
    return render_template(
        'dashboard.html', 
        page_title="Insight Dashboard",
        comparison_data=pd.DataFrame(model_comparison_data).to_dict('records')
    )

if __name__ == '__main__':
    # Create the model_artifacts directory if it doesn't exist
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    app.run(debug=True)