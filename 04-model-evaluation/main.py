import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Step 1: Load Clean Data
df = pd.read_csv("students_clean.csv")

print("Share :", df.shape)
print()
print(" First 5 Rows :")
print(df.head())
print()

#Feature (X) - the inputs. Skip 'name' (not predictive) and 'passed' (the answer)

feature_cols = ["study_hours", "attendance_pct", "math_score", "english_score"]
X = df[feature_cols]
Y= df["passed"]

print("Features (X) shape:", X.shape)
print("Label (y) shape:", Y.shape)
print()
print("Passed distribution:")
print(Y.value_counts())

# Step 2 : Split and Train

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.25, random_state = 42)

print("Training Students : ", len(X_train))
print("Testing Students :" , len(X_test))

print()

# Train a Logistic Regression Model (Classification, not regression)

model = LogisticRegression(max_iter = 1000)
model.fit(X_train,Y_train)

print("Model Trained.")

print()

# Make Predictions on test set
predictions = model.predict(X_test)

print("Predictions :", predictions)
print("Actual :  ", Y_test.values)

# Step 3 : Evaluation Metrics

# Accuracy
accuracy = accuracy_score(Y_test, predictions)
print("Accuracy :", accuracy)
print()

#Confision Matrix - where exactly did we get right/wrong
print("Confusion Matrix ")
print(confusion_matrix(Y_test, predictions))
print()

# Full Report - precision, recall

print("Classification Report :")
print(classification_report(Y_test, predictions, target_names = ["Failed","Passed"]))

# Step 4 : Train vs Test Accuracy

train_predictions = model.predict(X_train)
train_accuracy = accuracy_score(Y_train,train_predictions)

test_accuracy = accuracy_score(Y_test, predictions)

print(" Accuracy on Training Data :" , train_accuracy)
print(" Accuracy on Test Data :", test_accuracy)

print()

if train_accuracy > test_accuracy + 0.1:
    print("⚠️  Model may be OVERFITTING — much better on training than test")
else:
    print("✅ Model generalizes well — similar performance on both")
