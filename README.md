# Customer Feedback Analyser

Capstone project - Comparative Analysis of NLP and ML Approaches for Customer Feedback Classification.

## What it does

- **Classifier** - paste any review, get sentiment (Positive/Neutral/Negative) and complaint category instantly
- **Batch classification** - upload a CSV and classify thousands of reviews at once
- **Dashboard** - visual overview of the dataset (22,641 reviews)
- **Model Comparison** - all 7 models compared side by side
- **Insights** - actionable recommendations for management

## Models used

| Model | Sentiment F1 | Category F1 |
|---|---|---|
| Naive Bayes | 0.708 | 0.712 |
| Logistic Regression | 0.795 | 0.857 |
| Linear SVM | 0.800 | 0.912 |
| Random Forest | 0.726 | 0.848 |
| XGBoost | 0.783 | 0.964 |
| LightGBM | 0.795 | 0.970 |
| DistilBERT | 0.839 | - |




