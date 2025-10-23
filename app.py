# app.py - Flask Backend for Spam Classifier

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
from datetime import datetime

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

app = Flask(__name__)
CORS(app)

# Global variables
model = None
vectorizer = None
stop_words = set(stopwords.words('english'))
model_accuracy = 0.9698

# Load model and vectorizer
def load_model():
    global model, vectorizer
    try:
        if os.path.exists('model.pkl') and os.path.exists('vectorizer.pkl'):
            model = pickle.load(open('model.pkl', 'rb'))
            vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
            print("✓ Model and vectorizer loaded successfully")
            return True
        else:
            print("⚠ Model files not found. Train model first!")
            return False
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False

# Text preprocessing
def preprocess_text(text):
    """Clean and preprocess email text"""
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special chars
    text = ' '.join(text.split())  # Remove extra spaces
    return text

# Routes
@app.route('/api/health', methods=['GET'])
def health_check():
    """Check if API is running"""
    return jsonify({
        'status': 'online',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None
    })

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        'model_type': 'Naive Bayes Classifier',
        'accuracy': model_accuracy,
        'training_date': '2025-10-23',
        'dataset': 'SMS Spam Collection',
        'classes': ['spam', 'ham']
    })

@app.route('/api/predict', methods=['POST'])
def predict_single():
    """Classify single email as spam or ham"""
    try:
        data = request.json
        email_text = data.get('email', '').strip()

        if not email_text:
            return jsonify({
                'error': 'Email text is required',
                'success': False
            }), 400

        if len(email_text) < 3:
            return jsonify({
                'error': 'Email text too short',
                'success': False
            }), 400

        if model is None or vectorizer is None:
            return jsonify({
                'error': 'Model not loaded',
                'success': False
            }), 500

        # Preprocess
        cleaned_text = preprocess_text(email_text)

        # Vectorize
        email_vector = vectorizer.transform([cleaned_text])

        # Predict
        prediction = model.predict(email_vector)[0]
        confidence = max(model.predict_proba(email_vector)[0])
        probabilities = model.predict_proba(email_vector)[0]

        return jsonify({
            'success': True,
            'email': email_text[:100] + '...' if len(email_text) > 100 else email_text,
            'prediction': 'spam' if prediction == 1 else 'ham',
            'confidence': round(float(confidence), 4),
            'probability_spam': round(float(probabilities[1]), 4),
            'probability_ham': round(float(probabilities[0]), 4),
            'model_accuracy': model_accuracy
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/api/predict-batch', methods=['POST'])
def predict_batch():
    """Classify multiple emails"""
    try:
        data = request.json
        emails = data.get('emails', [])

        if not emails or not isinstance(emails, list):
            return jsonify({
                'error': 'emails list is required',
                'success': False
            }), 400

        if len(emails) > 100:
            return jsonify({
                'error': 'Maximum 100 emails per batch',
                'success': False
            }), 400

        if model is None or vectorizer is None:
            return jsonify({
                'error': 'Model not loaded',
                'success': False
            }), 500

        results = []
        spam_count = 0
        ham_count = 0

        for email in emails:
            if isinstance(email, str) and email.strip():
                cleaned_text = preprocess_text(email)
                email_vector = vectorizer.transform([cleaned_text])
                prediction = model.predict(email_vector)[0]
                confidence = max(model.predict_proba(email_vector)[0])
                pred_label = 'spam' if prediction == 1 else 'ham'

                if pred_label == 'spam':
                    spam_count += 1
                else:
                    ham_count += 1

                results.append({
                    'email': email[:50] + '...' if len(email) > 50 else email,
                    'prediction': pred_label,
                    'confidence': round(float(confidence), 4)
                })

        total = spam_count + ham_count
        spam_percentage = (spam_count / total * 100) if total > 0 else 0

        return jsonify({
            'success': True,
            'results': results,
            'spam_count': spam_count,
            'ham_count': ham_count,
            'total_processed': total,
            'spam_percentage': round(spam_percentage, 2)
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/api/spam-keywords', methods=['GET'])
def spam_keywords():
    """Return common spam indicators"""
    keywords = {
        'high_risk': ['urgent', 'click here', 'verify', 'confirm', 'act now', 'limited time', 'free', 'winner', 'claim'],
        'medium_risk': ['offer', 'deal', 'buy', 'save', 'discount', 'special'],
        'suspicious_patterns': ['all caps', 'multiple exclamation marks', 'multiple question marks']
    }
    return jsonify(keywords)

if __name__ == '__main__':
    if load_model():
        app.run(debug=True, port=5000)
    else:
        print("Cannot start app without trained model. Run model_trainer.py first.")