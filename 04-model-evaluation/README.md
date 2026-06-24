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

Features (X) shape: (20, 4)

Label (y) shape: (20,)
Passed distribution:

passed

1    12

0     8
Training Students: 15

Testing Students:  5

Model Trained.
Predictions: [1 1 0 0 1]

Actual:      [1 1 0 0 1]
Accuracy: 1.0
Confusion Matrix:

[[2 0]

[0 3]]
Classification Report:

precision    recall  f1-score   support

Failed       1.00      1.00      1.00         2

Passed       1.00      1.00      1.00         3

accuracy                           1.00         5
Accuracy on Training Data: 1.0

Accuracy on Test Data:     1.0

✅ Model generalizes well — similar performance on both

## Understanding the Metrics

- **Accuracy** — percentage of correct predictions overall
- **Confusion Matrix** — grid showing correct vs incorrect predictions per class.
  The diagonal is correct; off-diagonal is mistakes.
- **Precision** — when the model predicts "pass", how often is it right?
- **Recall** — of all students who actually passed, how many did the model catch?
- **F1-score** — single number combining precision and recall

## The Overfitting Check

The most important habit this project teaches:

> Always compare training accuracy to test accuracy.

- **Small gap** → model learned the real pattern → generalizes well
- **Large gap** (training ≫ test) → model memorized → overfitting

A model that scores 100% on training data but 65% on test data has memorized,
not learned. It will fail on new data in production.

## A Note on 100% Accuracy

This project hits 100% accuracy — but that's a **warning sign**, not a victory.
In real ML, 100% usually means:
- The test set is too small (only 5 students here)
- The data is artificially clean with an obvious pattern
- Data leakage (the answer leaked into the features)

Experienced engineers get suspicious of perfect scores, not excited by them.
The **technique** (comparing train vs test) is what matters, not the perfect number.

## What we learned

- Classification predicts categories; regression predicts numbers
- Logistic Regression outputs probabilities, ideal for yes/no outcomes
- Accuracy alone hides what kind of mistakes a model makes
- The confusion matrix shows where predictions go right and wrong
- Comparing train vs test accuracy is how you detect overfitting

## Next article

**05 — Build a Spam Email Classifier**: our first text-based project — turning
words into numbers a model can understand, then classifying messages as spam or not.