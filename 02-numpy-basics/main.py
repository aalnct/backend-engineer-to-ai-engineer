import numpy as np

# Python list
prices_list = [100,200,300,400]

# NumPy Array
prices_array = np.array([100,200,300,400])

prices_with_tax_list = [p * 1.1 for p in prices_list]
prices_with_tax_array = prices_array * 1.1

print("List :" , prices_with_tax_list )
print("Array :", prices_with_tax_array)

# Step 2 : 2D Arrays
houses = np.array([[1200,2,10], [1500,3,8],[1800,3,5],[2000,4,3]])

print("Shape :" , houses.shape)
print("Rows :" , houses.shape[0])
print("Cols :" , houses.shape[1])

print()

print("First House :" , houses[0])
print("All sizes :" , houses[:,0])
print("Bedrooms of house 2 :" , houses[1,1])

# Step 3 : Array Math ------

sizes = houses[:,0] # all sizes

# Basic Stats, no loop needed

print ("Min Size : ", np.min(sizes) )
print("Max Size : ", np.max(sizes))
print("Mean Size :", np.mean(sizes))
print("Std Dev : ", np.std(sizes))

print()

#Normalization -- scaling values to 0-1 range
# This is one of the most common pre-processing steps in ML
#Formula :(value - min)/(max - min)

min_size = np.min(sizes)
max_size = np.max(sizes)
normalized = (sizes - min_size) / (max_size - min_size)

print("Original sizes:   ", sizes)
print("Normalized sizes: ", normalized)

# Step 4 : Dot product

# Pretend these are the weights our model learned in Article 1
# one weight per feature: [size_sqft, bedrooms, age]

# Explaining example below

# 200 — each extra square foot adds $200 to the price
# 15000 — each extra bedroom adds $15,000
# -3000 — each extra year of age subtracts $3,000 (older = cheaper)

# in article 1, model found these weights automatically - model.fit()

weights = np.array([200,15000, -3000])

house = np.array([1500,3,8])

# Manual Calculation

manual = (house[0] * weights[0] + house[1] * weights[1] + house[2] * weights[2])

#Dot Product (numPy way)

dot_result = np.dot(house,weights)

print("Manual result: ", manual)
print("Dot product:   ", dot_result)
print("Same result?   ", manual == dot_result)
print()

# apply to all houses at once

all_predictions = np.dot(houses, weights)
print("Predictions for all houses:", all_predictions)


# ── Step 5: Useful Array Creation ───────────────────

# Zeros and ones — useful for initializing weights
zeros = np.zeros((3, 3))
ones  = np.ones((3, 3))

print("Zeros:\n", zeros)
print("Ones:\n", ones)
print()

# Random array — useful for testing shapes without real data
# seed = reproducible (like random_state=42 in Article 1)
np.random.seed(42)
random_matrix = np.random.rand(4, 3)

print("Random matrix shape:", random_matrix.shape)
print("Random matrix:\n", random_matrix)
print()

# Reshape — change dimensions without changing data
flat = np.array([1, 2, 3, 4, 5, 6])
reshaped = flat.reshape(2, 3)   # 2 rows, 3 columns
print("Flat:     ", flat)
print("Reshaped:\n", reshaped)