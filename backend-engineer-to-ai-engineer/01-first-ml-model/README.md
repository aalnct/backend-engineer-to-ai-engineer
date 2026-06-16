# 01 — From Java Backend to ML: Build Your First Python ML Model

**Project:** House price prediction using `pandas` + `scikit-learn`

This is the companion project for the first article in the *From Backend Engineer to AI Engineer* series.

## What this project does

1. Loads housing data (`house_prices.csv`) into a pandas DataFrame
2. Selects features (`size_sqft`, `bedrooms`, `age`) and a label (`price`)
3. Splits the data into training (80%) and testing (20%) sets
4. Trains a `LinearRegression` model
5. Predicts prices on the test set
6. Measures error using Mean Absolute Error (MAE)

## Setup

```bash
python3 -m venv venv

# Activate
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate          # Windows

pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Expected Output

```
Loaded data:
   size_sqft  bedrooms  age   price
0       1200         2   10  250000
1       1500         3    8  320000
2       1800         3    5  400000
3       2000         4    3  500000
4       2500         4    2  650000

Training rows: 16
Testing rows:  4

Predictions vs Actual:
  Predicted: $229,117.17   Actual: $250,000.00
  Predicted: $463,188.11   Actual: $460,000.00
  Predicted: $169,130.79   Actual: $200,000.00
  Predicted: $320,604.70   Actual: $320,000.00

Average Error (MAE): $13,886.21
```

## What we learned

- How to load tabular data with pandas (`read_csv`)
- The difference between features (`X`) and a label (`y`)
- Why we split data into training and testing sets
- How to train and use a `LinearRegression` model
- How to measure prediction error with Mean Absolute Error (MAE)

## Next article

**02 — NumPy Explained for Backend Engineers**: arrays, vectors, and basic matrix operations — the foundation that pandas and scikit-learn are built on.
