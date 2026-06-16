# 02 — NumPy Explained for Backend Engineers

**Project:** Understanding arrays, vectorization, and matrix math — the foundation every ML library is built on.

This is the companion project for Article 2 in the *From Backend Engineer to AI Engineer* series.

## What this project covers

1. NumPy arrays vs Python lists — and why the difference matters
2. 2D arrays (matrices) — shape, indexing, slicing
3. Array math — stats, normalization, vectorization
4. Dot product — the math behind `model.predict()`
5. Useful array creation — zeros, ones, random, reshape

## Why NumPy before pandas or scikit-learn?

Both pandas and scikit-learn are built on top of NumPy.
When you called `model.fit(X_train, y_train)` in Article 1,
`X_train` was a NumPy array under the hood.
Understanding NumPy makes everything else click.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate          # Windows

pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Expected Output

List : [110.00000000000001, 220.00000000000003, 330.0, 440.00000000000006]

Array : [110. 220. 330. 440.]

Shape : (4, 3)

Rows : 4

Cols : 3

First House : [1200    2   10]

All sizes : [1200 1500 1800 2000]

Bedrooms of house 2 : 3

Min Size :  1200

Max Size :  2000

Mean Size : 1625.0

Std Dev :  303.10889132455355

Original sizes:    [1200 1500 1800 2000]

Normalized sizes:  [0.    0.375 0.75  1.   ]

Manual result:  321000

Dot product:    321000

Same result?    True

Predictions for all houses: [240000 321000 390000 451000]

Zeros:

[[0. 0. 0.]

[0. 0. 0.]

[0. 0. 0.]]

Ones:

[[1. 1. 1.]

[1. 1. 1.]

[1. 1. 1.]]

Random matrix shape: (4, 3)

Random matrix:

[[0.37454012 0.95071431 0.73199394]

[0.59865848 0.15601864 0.15599452]

[0.05808361 0.86617615 0.60111501]

[0.70807258 0.02058449 0.96990985]]

Flat:      [1 2 3 4 5 6]

Reshaped:

[[1 2 3]

[4 5 6]]