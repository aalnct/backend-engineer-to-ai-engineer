import pandas as pd

# -- Step 1 : Load and Explore

df = pd.read_csv("students.csv")

print("Shape :" , df.shape)
print()
print("First 5 rows")
print(df.head())
print()
print("Column Type")
print(df.dtypes)
print()
print("Basic Stats :")
print(df.describe())

# - Step 2: Explore --
#Check for missing values - which columns, how many
print()
print("Missing values per column :")
print(df.isnull().sum())
print()

# See only rows with missing data
print("Rows with missing values :")
print(df[df.isnull().any(axis=1)])
print()

# Unique values in passed column

print("Unique values in passed :" , df["passed"].unique())
print("Value Counts")
print(df["passed"].value_counts())

print()

# Step 3 : Fix missing values

print("Mean Study hours (before fill) :", df["study_hours"].mean())
print()

# fill missing study hours with column mean

df["study_hours"] = df["study_hours"].fillna(df["study_hours"].mean())

# Verify - should show 0 missing now

print("Missing values after fill :")
print(df.isnull().sum())
print()

# Check Dave and Leo got filled

print("Dave and Leo after fill :")
print(df[df["name"].isin(["Dave","Leo"])])


# Step 4 : Convert text to numbers ----

#Map yes -> 1, no -> 0

df["passed"] = df["passed"].map({"yes" : 1, "no" : 0})

print()
print("After conversion")

print(df[["name", "passed"]].head(10))

print()

print("Column type now :" , df["passed"].dtype)
print()

#Quick Sanity Check

print(" Value counts after conversion")
print( df["passed"].value_counts()  )

# Step 5 : Select features and Analyse Correlations

features = ["study_hours", "attendance_pct", "math_score", "english_score"]

# Correlation with passed - how strongly does each feature relate ?
# 1.0 = perfect, -1.0 = perfect negative, 0 = no relationship

print("Correlation with 'passed': ")
print(df[features + ["passed"]].corr()["passed"].sort_values(ascending=False))

print()

# Average values for passed vs failed students
print("Average stats — passed vs failed:")
print(df.groupby("passed")[features].mean().round(2))


# Step 6 : Save the clean data

df.to_csv("students_clean.csv", index = False)
print(" Saved Student_clean.csv")

print()

# Verify by reloading and checking

verify =pd.read_csv("students_clean.csv")

print(" Reloaded Shape : ", verify.shape)
print("Column Types :")
print(verify.dtypes)

print()

print("First 3 rows of clean data :")
print(verify.head(3))