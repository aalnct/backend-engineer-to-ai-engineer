# 04 — Train/Test Split and Model Evaluation

**Project:** Train a classifier on the cleaned student data and evaluate it properly — accuracy, confusion matrix, precision/recall, and detecting overfitting.

This is the companion project for Article 4 in the *From Backend Engineer to AI Engineer* series.

## What this project covers

1. Loading cleaned data from Article 3 (`students_clean.csv`)
2. Regression vs Classification — why we use Logistic Regression here
3. Train/test split — holding back data to test honestly
4. Evaluation metrics — accuracy, confusion matrix, precision, recall, F1
5. Detecting overfitting — comparing training vs test accuracy

## Regression vs Classification

Article 1 used **Linear Regression** to predict a continuous number (house price).
This project predicts a category — passed (1) or failed (0) — which is **classification**.
The right tool is **Logistic Regression**, which outputs a probability between 0 and 1
instead of an unbounded number.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate          # Windows

pip install -r requirements.txt
```

This project needs `students_clean.csv` from Article 3. It's already included here,
but if you want to regenerate it, run the Article 3 project first.

## Run

```bash
python main.py
```

## Expected Output