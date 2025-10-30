from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from scipy.sparse import hstack  # for sparse matrices stashing
import re
from textblob import TextBlob
from flask import send_from_directory

app = Flask(__name__)
CORS(app)

# Load models and vectorizers
lr_model = joblib.load('lr_model.pkl')
nb_model = joblib.load('nb_model.pkl')
word_vectorizer = joblib.load('word_vectorizer.pkl')
char_vectorizer = joblib.load('char_vectorizer.pkl')

# dynamically recreating numeric values

def extract_numeric_features(text):
    # Recreateing the same numeric features used during training."""
    words = text.split()
    word_count = len(words)
    char_count = len(text)
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    exclams = text.count('!')
    questions = text.count('?')
    dots = text.count('.')
    uppercase = sum(1 for c in text if c.isupper())
    upper_ratio = uppercase / char_count if char_count > 0 else 0
    digit_ratio = sum(1 for c in text if c.isdigit()) / char_count if char_count > 0 else 0

    # These two depend on your preprocessing pipeline
    spam_keywords = ['free', 'win', 'winner', 'cash', 'urgent', 'prize', 'claim',
        'now', 'click', 'offer', 'congratulations', 'award', 'money']
    spam_keyword_hits = sum(1 for w in words if w.lower() in spam_keywords)

    # Sentiment using TextBlob
    sentiment = TextBlob(text).sentiment.polarity

    # Keep same order as in training
    return np.array([
        word_count, char_count, avg_word_len, exclams,
        questions, dots, uppercase, upper_ratio,
        digit_ratio, spam_keyword_hits, sentiment
    ]).reshape(1, -1)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    message = data.get('message')

    if not message:
        return jsonify({'error': 'Message is required'}), 400

    try:
        # Transform input using both vectorizers
        X_word = word_vectorizer.transform([message])
        X_char = char_vectorizer.transform([message])

        X_numeric = extract_numeric_features(message)

        # Combine features horizontally
        X_input = hstack([X_word, X_char, X_numeric])

        # Predictions
        lr_pred = lr_model.predict(X_input)[0]
        nb_pred = nb_model.predict(X_input)[0]

        # Confidence scores
        lr_conf = float(np.max(lr_model.predict_proba(X_input)[0]))
        nb_conf = float(np.max(nb_model.predict_proba(X_input)[0]))

    except Exception as e:
        print("🔥 Error during prediction:", e)
        return jsonify({'error': str(e)}), 500

    return jsonify({
        'logistic_regression': {'label': str(lr_pred), 'confidence': lr_conf},
        'naive_bayes': {'label': str(nb_pred), 'confidence': nb_conf}
    })


@app.route("/", methods=["GET"])
def serve_root():
    return send_from_directory("static", "SpamlyGroup3.html")

@app.route("/spamly.js", methods=["GET"])
def serve_js():
    return send_from_directory("static", "spamly.js")


if __name__ == '__main__':
    app.run(debug=True)
