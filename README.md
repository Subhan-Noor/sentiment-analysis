# Sentiment Analysis on FCC Comments Using SVM

A Python-based sentiment analysis system built to classify and interpret public opinion regarding the repeal of net neutrality. This project uses classic machine learning methods — TF-IDF, dimensionality reduction, and an optimized SVM — to analyze comment sentiment at scale.

## 🚀 Features

- Trains an SVM classifier on preprocessed textual data
- Saves model + transformers (TF-IDF, SVD) for future predictions
- Predicts sentiment on unseen comment data
- Generates interpretive visuals (word clouds, distributions, etc.)

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sentiment-analysis-svm.git
cd sentiment-analysis
```

2. Set up environment:
```bash
pip install -r requirements.txt
python -m nltk.downloader stopwords
```

## 📊 Training
To train the model on labeled data:
```bash
python scripts/sentiment_analysis.py --input data/deidentified_survey_results.csv
```
This creates:
- `outputs/svm_model.joblib`
- `outputs/tfidf_vectorizer.joblib`
- `outputs/svd_transformer.joblib`
- Evaluation reports and confusion matrix

## 📈 Insights & Visualizations
Open the notebooks in `notebooks/` to:
- Analyze predicted class distribution
- Generate word clouds for predicted sentiments
- View prediction confidence histograms

You can create your own input file with a single column `short_comment`.
