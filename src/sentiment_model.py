import os
import pandas as pd
import numpy as np
import re
import joblib
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Make sure NLTK stopwords are downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase and split into words
    text = text.lower().strip()
    return text

def train_sentiment_model(data_path="data/imdb_reviews/IMDB Dataset.csv", model_dir="models"):
    start_time = time.time()
    print("Step 1: Loading IMDb Reviews Dataset...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"IMDb reviews dataset not found at {data_path}")
        
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} reviews.")
    
    # Check shape and sample
    print(df["sentiment"].value_counts())
    
    # 2. Text Preprocessing
    print("Step 2: Preprocessing text reviews (removing HTML, special characters, lowercasing)...")
    df["clean_review"] = df["review"].apply(clean_text)
    
    # Convert sentiment labels to binary (1 for positive, 0 for negative)
    df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})
    
    # Split into train and test
    print("Splitting dataset into train and test sets (80/20 split)...")
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_review"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )
    
    # 3. Vectorization (TF-IDF)
    print("Step 3: Creating TF-IDF Vectorizer...")
    stop_words_list = list(stopwords.words('english'))
    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words=stop_words_list,
        ngram_range=(1, 2) # Use unigrams and bigrams
    )
    
    print("Fitting vectorizer on training data...")
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    
    # Make sure output directory exists
    os.makedirs(model_dir, exist_ok=True)
    
    # 4. Model Training & Evaluation
    print("Step 4: Training sentiment models...")
    
    # Model A: Naive Bayes (Baseline)
    print("Training Multinomial Naive Bayes...")
    nb_model = MultinomialNB()
    nb_model.fit(X_train_vectorized, y_train)
    y_pred_nb = nb_model.predict(X_test_vectorized)
    acc_nb = accuracy_score(y_test, y_pred_nb)
    print(f"Naive Bayes Accuracy: {acc_nb:.4f}")
    print(classification_report(y_test, y_pred_nb, target_names=["Negative", "Positive"]))
    
    # Model B: Logistic Regression (State of the art for linear text classification)
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
    lr_model.fit(X_train_vectorized, y_train)
    y_pred_lr = lr_model.predict(X_test_vectorized)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    print(f"Logistic Regression Accuracy: {acc_lr:.4f}")
    print(classification_report(y_test, y_pred_lr, target_names=["Negative", "Positive"]))
    
    # Determine the best model
    if acc_lr >= acc_nb:
        best_model = lr_model
        best_name = "Logistic Regression"
        best_acc = acc_lr
    else:
        best_model = nb_model
        best_name = "Naive Bayes"
        best_acc = acc_nb
        
    print(f"Best model chosen: {best_name} with {best_acc:.4f} accuracy.")
    
    # Save the model and vectorizer
    print(f"Saving vectorizer and model to {model_dir}...")
    joblib.dump(vectorizer, os.path.join(model_dir, "tfidf_vectorizer.pkl"))
    joblib.dump(best_model, os.path.join(model_dir, "sentiment_model.pkl"))
    
    # Save a small metadata file
    import json
    metadata = {
        "model_name": best_name,
        "accuracy": best_acc,
        "vocab_size": len(vectorizer.vocabulary_),
        "training_time_seconds": time.time() - start_time
    }
    with open(os.path.join(model_dir, "sentiment_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
        
    print("Sentiment model training pipeline completed successfully!")

class SentimentPredictor:
    def __init__(self, model_dir="models"):
        self.vectorizer = joblib.load(os.path.join(model_dir, "tfidf_vectorizer.pkl"))
        self.model = joblib.load(os.path.join(model_dir, "sentiment_model.pkl"))
        
    def predict(self, text):
        cleaned = clean_text(text)
        vectorized = self.vectorizer.transform([cleaned])
        pred = self.model.predict(vectorized)[0]
        prob = self.model.predict_proba(vectorized)[0]
        
        # Sentiment mapping (1 = positive, 0 = negative)
        sentiment_label = "positive" if pred == 1 else "negative"
        confidence = prob[pred]
        
        return {
            "sentiment": sentiment_label,
            "confidence": float(confidence),
            "score": float(prob[1] - prob[0]) # Ranges from -1.0 (most negative) to 1.0 (most positive)
        }

if __name__ == "__main__":
    train_sentiment_model()
