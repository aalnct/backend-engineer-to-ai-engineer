"""
Article 1: From Java Backend to ML — Build Your First Python ML Model

Project: House price prediction using pandas + scikit-learn.

This script:
1. Loads housing data from a CSV
2. Selects features (X) and label (y)
3. Splits data into training and testing sets
4. Trains a Linear Regression model
5. Makes predictions on test data
6. Measures prediction error
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


def load_data(csv_path: str) -> pd.DataFrame:
    """Load housing data from CSV into a pandas DataFrame."""
    return pd.read_csv(csv_path)


def main():
    # 1. Load data
    data = load_data("house_prices.csv")
    print("Loaded data:")
    print(data.head())
    print()

    # 2. Select features (X) and label (y)
    #    Features: the inputs the model uses to make a prediction
    #    Label: the value we want the model to predict
    X = data[["size_sqft", "bedrooms", "age"]]
    y = data["price"]

    # 3. Split data into training and testing sets
    #    80% for training, 20% for testing
    #    random_state ensures the split is reproducible
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows:  {len(X_test)}")
    print()

    # 4. Train a Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 5. Make predictions on the test set
    predictions = model.predict(X_test)

    # 6. Measure error — Mean Absolute Error (MAE)
    #    MAE = average of |actual - predicted| across all test rows
    error = mean_absolute_error(y_test, predictions)

    print("Predictions vs Actual:")
    for pred, actual in zip(predictions, y_test):
        print(f"  Predicted: ${pred:,.2f}   Actual: ${actual:,.2f}")
    print()
    print(f"Average Error (MAE): ${error:,.2f}")


if __name__ == "__main__":
    main()
