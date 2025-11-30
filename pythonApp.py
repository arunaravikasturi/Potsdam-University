#!/usr/bin/env python3

import pandas as pd
import numpy as np
import statsmodels.api as sm

# ------------------------------------------------------------------
# Load CSV file
# ------------------------------------------------------------------
csv_path = "dataset01.csv"   # assumes file is in /tmp
df = pd.read_csv(csv_path)

# Make sure column 'y' exists
if 'y' not in df.columns:
    raise ValueError("Column 'y' not found in dataset01.csv")

# If 'x' is the single influence factor:
if 'x' not in df.columns:
    raise ValueError("Column 'x' not found in dataset01.csv")

y = df['y']
X = df[['x']]   # X as 2D (DataFrame), not 1D Series

# ------------------------------------------------------------------
# 1) Number of data entries of column 'y'
# ------------------------------------------------------------------
n_entries = y.shape[0]
print(f"Number of entries in column 'y': {n_entries}")

# ------------------------------------------------------------------
# 2) Mean of 'y'
# ------------------------------------------------------------------
mean_y = y.mean()
print(f"Mean of 'y': {mean_y:.4f}")

# ------------------------------------------------------------------
# 3) Standard deviation of 'y'
# ------------------------------------------------------------------
std_y = y.std(ddof=1)  # sample std dev
print(f"Standard deviation of 'y': {std_y:.4f}")

# ------------------------------------------------------------------
# 4) Variance of 'y'
# ------------------------------------------------------------------
var_y = y.var(ddof=1)  # sample variance
print(f"Variance of 'y': {var_y:.4f}")

# ------------------------------------------------------------------
# 5) Min and max of 'y'
# ------------------------------------------------------------------
min_y = y.min()
max_y = y.max()
print(f"Min of 'y': {min_y}")
print(f"Max of 'y': {max_y}")

# ------------------------------------------------------------------
# 6) OLS model: y ~ x
# ------------------------------------------------------------------
# Add intercept (constant term) to X
X_with_const = sm.add_constant(X)

model = sm.OLS(y, X_with_const)
results = model.fit()

print("\n=== OLS Regression Results ===")
print(results.summary())

# ------------------------------------------------------------------
# Store OLS model in a file called 'OLS_model'
# ------------------------------------------------------------------
# statsmodels results objects can be saved via .save()
results.save("OLS_model")

print("\nOLS model saved to file 'OLS_model'.")
