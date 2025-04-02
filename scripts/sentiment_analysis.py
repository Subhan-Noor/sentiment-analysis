# sentiment_final_svm_only.py

import pandas as pd
import numpy as np
import argparse
import logging
import os
import re
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)

def load_data(filepath):
    logging.info(f"Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    df = df.dropna(subset=['short_comment', 'pro_repeal'])
    df['pro_repeal'] = df['pro_repeal'].astype(int)
    return df

def preprocess_text(text):
    text = re.sub(r'[^A-Za-z\s]', '', str(text)).lower()
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(text)
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

def preprocess_column(df, column='short_comment'):
    logging.info("Preprocessing text column...")
    df[column] = df[column].apply(preprocess_text)
    return df

def vectorize_and_reduce_fit(df, column='short_comment', n_components=5):
    logging.info("Fitting TF-IDF and SVD on training data...")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 1), min_df=2, max_df=0.85)
    X_tfidf = tfidf.fit_transform(df[column])
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_reduced = svd.fit_transform(X_tfidf)
    return X_reduced, tfidf, svd

def train_svm(X_train, y_train):
    logging.info("Training SVM model...")
    svm_model = SVC(C=0.01, kernel='linear', probability=True, random_state=42)
    svm_model.fit(X_train, y_train)
    return svm_model

def evaluate_model(model, X_test, y_test, df_test_texts, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    logging.info("Evaluating SVM model...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)

    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    with open(os.path.join(output_dir, "report_svm.txt"), "w") as f:
        f.write(report)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - SVM')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix_svm.png"))
    plt.close()

    # Save predictions for later insights
    preds_df = pd.DataFrame({
        'short_comment': df_test_texts,
        'true_label': y_test,
        'predicted_label': y_pred,
        'score': y_prob
    })
    preds_df.to_csv(os.path.join(output_dir, "svm_predictions.csv"), index=False)

    metrics_summary = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'AUC-ROC': roc_auc_score(y_test, y_prob)
    }

    print("\nSVM Model Performance:")
    for k, v in metrics_summary.items():
        print(f"{k}: {v:.4f}")

    return metrics_summary

def save_model(model, tfidf, svd, output_dir="outputs"):
    joblib.dump(model, os.path.join(output_dir, "svm_model.joblib"))
    joblib.dump(tfidf, os.path.join(output_dir, "tfidf_vectorizer.joblib"))
    joblib.dump(svd, os.path.join(output_dir, "svd_transformer.joblib"))
    logging.info("Model and transformers saved to outputs/.")

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate SVM sentiment classifier")
    parser.add_argument('--input', type=str, required=True, help="Path to the input CSV file")
    args = parser.parse_args()

    df = load_data(args.input)
    df = preprocess_column(df, 'short_comment')

    X, tfidf, svd = vectorize_and_reduce_fit(df, 'short_comment', n_components=5)
    y = df['pro_repeal']

    X_train, X_test, y_train, y_test, text_train, text_test = train_test_split(
        X, y, df['short_comment'], test_size=0.2, random_state=42)

    model = train_svm(X_train, y_train)
    evaluate_model(model, X_test, y_test, text_test)
    save_model(model, tfidf, svd)

if __name__ == "__main__":
    main()
