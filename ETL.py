# ============================================================
#  ETL Pipeline: Data Preprocessing & Feature Engineering
#  For sentiment analysis task with 12 model variants
# ============================================================

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


# ── 1. Basic Text Cleaning ───────────────────────────────────

def clean_text(text):
    """Remove URLs, mentions, special chars, convert to lowercase."""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)  # URLs
    text = re.sub(r'@\w+', '', text)  # mentions
    text = re.sub(r'[^a-z\s]', '', text)  # special chars
    text = re.sub(r'\s+', ' ', text).strip()  # extra spaces
    return text


# ── 2. Three Preprocessing Schemes ───────────────────────────

def preprocess_scheme_1(texts):
    """
    Scheme 1: Clean + lowercase (minimal preprocessing).
    Returns tokenized text.
    """
    return [clean_text(t).split() for t in texts]


def preprocess_scheme_2(texts):
    """
    Scheme 2: Clean + stopword removal + stemming.
    """
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    processed = []
    for t in texts:
        tokens = clean_text(t).split()
        tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
        processed.append(tokens)
    return processed


def preprocess_scheme_3(texts):
    """
    Scheme 3: Clean + stopword removal + lemmatization.
    """
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    processed = []
    for t in texts:
        tokens = clean_text(t).split()
        tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
        processed.append(tokens)
    return processed


PREPROCESSING_SCHEMES = {
    "scheme_1": preprocess_scheme_1,
    "scheme_2": preprocess_scheme_2,
    "scheme_3": preprocess_scheme_3,
}


# ── 3. Feature Extraction ────────────────────────────────────

def extract_bow_features(texts, max_features=500):
    """Bag-of-Words using CountVectorizer."""
    vectorizer = CountVectorizer(max_features=max_features, lowercase=True)
    features = vectorizer.fit_transform(texts)
    return features, vectorizer


def extract_tfidf_features(texts, max_features=500):
    """TF-IDF features using TfidfVectorizer."""
    vectorizer = TfidfVectorizer(max_features=max_features, lowercase=True)
    features = vectorizer.fit_transform(texts)
    return features, vectorizer


# ── 4. Helper: Detokenize for vectorizers ─────────────────────

def detokenize(token_lists):
    """Convert list of token lists back to space-separated strings."""
    return [' '.join(tokens) for tokens in token_lists]


# ── 5. Model Configuration Factory ───────────────────────────

def get_12_model_configs():
    """
    Return 12 model configurations:
    - 3 preprocessing schemes
    - 2 feature types (BoW, TF-IDF)
    - 2 classifiers (Naive Bayes, Logistic Regression)
    Total: 3 × 2 × 2 = 12 models
    """
    configs = []
    classifiers = ["naive_bayes", "logistic_regression"]
    feature_types = ["bow", "tfidf"]
    
    for scheme_name in ["scheme_1", "scheme_2", "scheme_3"]:
        for feature_type in feature_types:
            for clf in classifiers:
                config_name = f"{scheme_name}_{feature_type}_{clf}"
                configs.append({
                    "name": config_name,
                    "preprocessing_scheme": scheme_name,
                    "feature_type": feature_type,
                    "classifier": clf,
                })
    
    return configs