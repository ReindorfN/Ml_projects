# Movie Recommendation System

A comprehensive machine learning project that analyzes the MovieLens dataset to extract business insights and deliver a content-based movie recommendation system. This project was developed as part of a Machine Learning course final assessment, where we acted as Data Scientists/ML Engineers for a leading streaming platform.

## Project Overview

This project transforms raw customer interaction data from the MovieLens dataset into actionable business insights and a deployable recommendation system. The system leverages machine learning models to predict movie ratings and provide personalized recommendations based on movie features such as genres and popularity metrics.

### Dataset

The MovieLens dataset contains:
- **100,836** ratings from **610** users
- **9,742** movies across multiple genres
- **3,683** tag applications
- **1,128** genome tags with relevance scores
- Data spanning from March 29, 1996 to September 24, 2018

## Data Analysis & Business Insights

### Data Understanding & Pre-Processing

The project began with comprehensive data exploration and cleaning:

- **Data Loading**: Processed 6 datasets (ratings, movies, tags, links, genome tags, genome scores)
- **Data Quality Assessment**: Identified and handled missing values, duplicates, and inconsistencies
- **Feature Engineering**: 
  - Transformed timestamps to extract temporal patterns (year, month, day of week)
  - Parsed pipe-separated genre strings into structured format
  - Created one-hot encoded genre features (19 unique genres)
  - Calculated log-transformed rating counts as popularity indicators
- **Data Filtering**: Applied quality threshold (minimum 50 ratings per movie) to ensure robust predictions

### Business Insights & Visual Analytics

The analysis uncovered several key insights across three main categories:

#### User Behavior Insights

1. **Rating Bias Analysis**: Identified distinct user segments including harsh critics (users who consistently rate below average) and generous raters (users who rate above average), enabling targeted content strategies.

2. **Temporal Trends**: Analyzed rating evolution over time, revealing that rating activity peaked in the mid-2000s while maintaining stable quality ratings across different time periods.

3. **User Retention Analysis**: Examined user activity patterns and engagement over time, identifying retention challenges and opportunities for improving user engagement.

#### Content Insights

4. **Genre Performance Analysis**: Discovered that while Drama dominates in quantity, niche genres often show higher quality ratings, suggesting opportunities for content diversification.

5. **Tag Sentiment Analysis**: Explored correlations between user-generated tags and ratings, revealing how tag sentiment relates to movie quality perception.

6. **Tag Clustering**: Identified thematic clusters such as "Mind-Bending", "Hidden Gems", and other content categories that help understand user preferences beyond traditional genres.

#### Hidden Patterns

7. **Release Year Impact**: Analyzed how movie age correlates with ratings, providing insights into temporal preferences and content longevity.

8. **Hidden Gems Identification**: Discovered 100+ high-quality movies with low visibility (high ratings but few ratings), presenting opportunities for targeted promotion and content discovery.

9. **User Taste Profile Clustering**: Used K-Means clustering to segment users into 5 distinct taste profiles, enabling personalized marketing and recommendation strategies.

## Machine Learning Models

### Model Training & Evaluation

Three regression models were trained to predict continuous movie ratings (0.5 to 5.0 scale) using movie features:

#### Features Used
- **19 Genre Features**: One-hot encoded binary indicators for each genre (Action, Adventure, Animation, Children, Comedy, Crime, Documentary, Drama, Fantasy, Film-Noir, Horror, IMAX, Musical, Mystery, Romance, Sci-Fi, Thriller, War, Western)
- **Popularity Feature**: Log-transformed rating count to normalize movie popularity

#### Models Trained

1. **Random Forest Regressor** 🏆
   - **Test RMSE**: 0.3965
   - **Test MAE**: 0.3075
   - **R² Score**: 0.3085
   - **Status**: Best performing model
   - **Characteristics**: Handles non-linear relationships, robust to overfitting

2. **Ridge Regression**
   - **Test RMSE**: 0.4039
   - **Test MAE**: 0.3133
   - **R² Score**: 0.2824
   - **Status**: Second best, fast and interpretable
   - **Characteristics**: Linear model with L2 regularization, provides feature importance insights

3. **K-Nearest Neighbors (KNN)**
   - **Test RMSE**: 0.4118
   - **Test MAE**: 0.3210
   - **R² Score**: 0.1324
   - **Status**: Good for similarity-based recommendations
   - **Characteristics**: Instance-based learning, useful for finding similar movies

#### Additional Model

4. **Logistic Regression** (Binary Classification)
   - **Accuracy**: ~85%
   - **Purpose**: Classifies movies as "thumbs up" (rating ≥ 3.5) or "thumbs down"
   - **Use Case**: Binary recommendation filtering

### Model Performance Summary

The Random Forest model achieved the best performance with:
- **RMSE of 0.3965**: Indicates predictions are within approximately 0.4 rating points on average
- **MAE of 0.3075**: Average absolute error of about 0.31 rating points
- **R² of 0.3085**: Explains about 31% of variance in ratings, a significant improvement over naive baseline (global mean ≈ 3.5)

All models significantly outperformed the baseline (global average rating), demonstrating that machine learning can effectively predict movie quality using content features alone.

## Flask Web Application

A user-friendly web interface was developed to make the recommendation system accessible and interactive. The Flask application provides multiple features:

### Core Features

1. **Top-N Recommendations**
   - Displays the highest-rated movie predictions based on the trained Random Forest model
   - Shows top 15 movies with predicted ratings, genres, and metadata
   - Provides "Find Similar" functionality for each recommended movie

2. **Similar Movies Discovery**
   - Uses K-Nearest Neighbors (KNN) algorithm to find movies similar to a selected movie
   - Based on cosine similarity in the feature space (genres + popularity)
   - Features autocomplete search for easy movie selection
   - Displays similarity scores and genre information

3. **Genre-Based Recommendations**
   - Allows users to filter top recommendations by specific genres
   - Returns top 10 highest-rated movies in the selected genre
   - Highlights the selected genre in movie genre tags
   - Enables discovery of quality content within preferred genres

4. **Insight Dashboard**
   - **Model Performance Comparison**: Visual comparison of all three regression models with RMSE and MAE metrics
   - **Data Visualizations**: 
     - User rating distribution showing overall sentiment patterns
     - Rating trends over time revealing temporal patterns
     - User rating bias analysis identifying rating behavior patterns
   - **Dataset Statistics**: Summary of models trained, genre features, and qualified movies
   - **Methodology & Conclusions**: Detailed explanation of the approach and key findings

### Technical Implementation

- **Backend**: Flask web framework with scikit-learn models
- **Frontend**: Responsive HTML/CSS with modern UI design
- **Model Serving**: Pre-trained models loaded from pickle files for fast inference
- **Similarity Search**: KNN model using cosine similarity for content-based recommendations
- **Data Processing**: Efficient pandas operations for filtering and ranking

### User Experience

The application provides an intuitive interface where users can:
- Browse top-rated movie recommendations
- Search for movies and discover similar content
- Explore recommendations by genre preference
- View comprehensive analytics and model performance metrics

## Key Achievements

- ✅ **Comprehensive Data Pipeline**: End-to-end preprocessing from raw CSV files to model-ready features
- ✅ **15+ Analytical Functions**: Extensive business intelligence with visualizations
- ✅ **Multiple ML Models**: Trained and evaluated 4 different algorithms
- ✅ **Production-Ready Web App**: Deployable Flask application with multiple recommendation features
- ✅ **Business Value**: Identified actionable insights for content strategy and user engagement

## Technologies Used

- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn (Random Forest, Ridge Regression, KNN, Logistic Regression)
- **Web Framework**: Flask
- **Visualization**: matplotlib, seaborn (for analysis notebooks)
- **Model Persistence**: pickle

## Project Impact

This project demonstrates how machine learning can transform raw user interaction data into:
- **Actionable Business Insights**: User segmentation, content performance analysis, and hidden pattern discovery
- **Deployable Recommendation System**: A working web application that provides personalized movie recommendations
- **Scalable Architecture**: Modular design that can be extended with additional features and models

The system successfully predicts movie ratings using only content features (genres and popularity), making it valuable for scenarios where user history is unavailable or for new user onboarding.
