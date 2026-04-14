# ============================================================
#  Task 3 – Machine Learning Models (12 Variants)
#  Preprocessing Schemes × Feature Types × Classifiers
#  Uses labeled_data.csv from task3 script
# ============================================================

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import warnings

from ETL import (
    PREPROCESSING_SCHEMES,
    extract_bow_features,
    extract_tfidf_features,
    detokenize,
    get_12_model_configs,
)

warnings.filterwarnings('ignore')


# ── Configuration ────────────────────────────────────────────

OUTPUT_PATH = "model_results_12.csv"
LABELED_DATA_PATH = "labeled_data.csv"
TRAIN_TEST_SPLIT = 0.3
RANDOM_STATE = 42


# ── 1. Load labeled data ─────────────────────────────────────

def load_ground_truth():
    """Load labeled_data.csv from task3 output."""
    if not os.path.exists(LABELED_DATA_PATH):
        raise FileNotFoundError(f"{LABELED_DATA_PATH} not found. Run task3.py first.")
    
    df = pd.read_csv(LABELED_DATA_PATH)
    print(f"Loaded {len(df)} labeled records from {LABELED_DATA_PATH}")
    print(f"Columns: {df.columns.tolist()}\n")
    
    # Identify text column and label column
    text_col = None
    for col in ["full_text_clean", "comment_body", "text", "title"]:
        if col in df.columns:
            text_col = col
            break
    
    label_col = "ground_truth" if "ground_truth" in df.columns else None
    
    if not text_col or not label_col:
        raise ValueError(f"Cannot identify text or label column in {LABELED_DATA_PATH}")
    
    return df, text_col, label_col


# ── 2. Build and evaluate one model ──────────────────────────

def train_and_evaluate_model(X_train, X_test, y_train, y_test, classifier_type):
    """
    Train classifier on X_train and evaluate on X_test.
    X_train, X_test are sparse matrices (BoW or TF-IDF).
    """
    if classifier_type == "naive_bayes":
        clf = MultinomialNB()
    elif classifier_type == "logistic_regression":
        clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Unknown classifier: {classifier_type}")
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "y_pred": y_pred,
        "clf": clf,
    }


# ── 3. Run all 12 models ─────────────────────────────────────

def run_all_12_models(df, text_col, label_col):
    """
    Train and evaluate all 12 model configurations.
    Return results dataframe.
    """
    
    # Get ground truth labels
    y = df[label_col].values
    texts = df[text_col].values
    
    # Train-test split
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        texts, y, test_size=TRAIN_TEST_SPLIT, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"Train set: {len(X_train_texts)} records")
    print(f"Test set: {len(X_test_texts)} records\n")
    
    configs = get_12_model_configs()
    results = []
    
    for idx, config in enumerate(configs, 1):
        print(f"[{idx}/12] Training model: {config['name']}")
        
        # Apply preprocessing scheme
        preprocess_fn = PREPROCESSING_SCHEMES[config['preprocessing_scheme']]
        X_train_tokens = preprocess_fn(X_train_texts)
        X_test_tokens = preprocess_fn(X_test_texts)
        
        # Detokenize back to strings for vectorizers
        X_train_str = detokenize(X_train_tokens)
        X_test_str = detokenize(X_test_tokens)
        
        # Extract features
        if config['feature_type'] == 'bow':
            X_train_features, vectorizer = extract_bow_features(X_train_str)
            X_test_features = vectorizer.transform(X_test_str)
        else:  # tfidf
            X_train_features, vectorizer = extract_tfidf_features(X_train_str)
            X_test_features = vectorizer.transform(X_test_str)
        
        # Train and evaluate classifier
        eval_result = train_and_evaluate_model(
            X_train_features, X_test_features,
            y_train, y_test,
            config['classifier']
        )
        
        result_row = {
            "model_id": idx,
            "model_name": config['name'],
            "preprocessing": config['preprocessing_scheme'],
            "feature_type": config['feature_type'],
            "classifier": config['classifier'],
            "accuracy": eval_result['accuracy'],
            "f1_macro": eval_result['f1_macro'],
        }
        results.append(result_row)
        
        print(f"  → Accuracy: {eval_result['accuracy']:.4f}, F1: {eval_result['f1_macro']:.4f}\n")
    
    return pd.DataFrame(results)


# ── 4. Main execution ────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("Task 3 – Machine Learning: 12 Model Configurations")
    print("=" * 70 + "\n")
    
    # Load labeled data
    df, text_col, label_col = load_ground_truth()
    
    # Run all 12 models
    results_df = run_all_12_models(df, text_col, label_col)
    
    # Save results
    results_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✓ Saved model results to: {OUTPUT_PATH}\n")
    
    # Summary statistics
    print("=" * 70)
    print("12-MODEL SUMMARY")
    print("=" * 70)
    print(results_df.to_string(index=False))
    
    print("\nBest Models (by Accuracy):")
    top_3 = results_df.nlargest(3, 'accuracy')[['model_name', 'accuracy', 'f1_macro']]
    print(top_3.to_string(index=False))
    
    print(f"\nAverage Accuracy: {results_df['accuracy'].mean():.4f}")
    print(f"Average F1-Macro: {results_df['f1_macro'].mean():.4f}")
    print("=" * 70)
