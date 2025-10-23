# model_trainer.py - Train spam classifier model

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Clean and preprocess text"""
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special chars
    text = ' '.join(text.split())  # Remove extra spaces
    return text

def load_data():
    """Load and prepare dataset"""
    try:
        # Try loading from existing CSV
        df = pd.read_csv('spam_dataset.csv', encoding='latin-1')
        
        # Handle different column names
        if 'v1' in df.columns and 'v2' in df.columns:
            df.columns = ['label', 'message']
        elif 'Label' in df.columns and 'Text' in df.columns:
            df.columns = ['label', 'message']
        
        print(f"✓ Dataset loaded: {len(df)} messages")
        print(f"  Spam: {len(df[df['label'] == 'spam'])}")
        print(f"  Ham: {len(df[df['label'] == 'ham'])}")
        
        return df
    
    except FileNotFoundError:
        print("⚠ Dataset not found. Creating synthetic dataset...")
        # Create synthetic dataset if file not found
        synthetic_data = {
            'label': ['ham'] * 50 + ['spam'] * 50,
            'message': [
                # Ham messages
                'Hi, how are you doing today?',
                'Meeting scheduled for tomorrow at 10 AM',
                'Thanks for the update',
                'Can you send me the report?',
                'Looking forward to our call',
                # Spam messages
                'CLICK HERE to win free iPhone!!!',
                'Urgent: Verify your account immediately',
                'Limited time offer - 50% off everything',
                'You are a WINNER! Claim your prize now',
                'Free money waiting for you'
            ] * 10  # Repeat to create 50 of each
        }
        return pd.DataFrame(synthetic_data)

def train_model():
    """Train Naive Bayes spam classifier"""
    print("\n" + "="*60)
    print("EMAIL SPAM CLASSIFIER - MODEL TRAINING")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Preprocess messages
    print("\n→ Preprocessing text...")
    df['cleaned_message'] = df['message'].apply(preprocess_text)
    
    # Convert labels to binary
    df['label_encoded'] = (df['label'] == 'spam').astype(int)
    
    # Split data
    print("→ Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_message'],
        df['label_encoded'],
        test_size=0.2,
        random_state=42,
        stratify=df['label_encoded']
    )
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    # Vectorize text
    print("\n→ Vectorizing text (TF-IDF)...")
    vectorizer = TfidfVectorizer(
        max_features=3000,
        min_df=2,
        max_df=0.95,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"  Vocabulary size: {len(vectorizer.get_feature_names_out())}")
    print(f"  Training matrix shape: {X_train_vec.shape}")
    
    # Train Naive Bayes
    print("\n→ Training Naive Bayes classifier...")
    model = MultinomialNB(alpha=1.0)
    model.fit(X_train_vec, y_train)
    
    # Predictions
    print("\n→ Evaluating model...")
    y_pred = model.predict(X_test_vec)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Results
    print("\n" + "="*60)
    print("MODEL PERFORMANCE")
    print("="*60)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score:  {f1:.4f}")
    
    print("\n" + "="*60)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
    
    print("="*60)
    print("CONFUSION MATRIX")
    print("="*60)
    cm = confusion_matrix(y_test, y_pred)
    print(f"True Negatives:  {cm[0][0]}")
    print(f"False Positives: {cm[0][1]}")
    print(f"False Negatives: {cm[1][0]}")
    print(f"True Positives:  {cm[1][1]}")
    
    # Save model
    print("\n→ Saving model and vectorizer...")
    pickle.dump(model, open('model.pkl', 'wb'))
    pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))
    
    print("✓ Model saved to model.pkl")
    print("✓ Vectorizer saved to vectorizer.pkl")
    
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETE - Ready to start backend!")
    print("→ Run: python app.py")
    print("="*60 + "\n")
    
    return model, vectorizer, accuracy

if __name__ == '__main__':
    train_model()