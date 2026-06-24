"""
Data Visualization Script
-----------------------------------
Generates visualizations for all variables in a dataset.

Outputs:
- Missing data heatmap
- Numeric variable distributions
- Boxplots
- Categorical variable bar plots
- Correlation heatmap
- Pairplots

All plots saved to 'EDA_Visualizations' folder
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -----------------------------
# Load Dataset
# -----------------------------
df = ml_data.copy()

# -----------------------------
# Create output directory
# -----------------------------
output_dir = "Outputs/EDA_Visualizations"
os.makedirs(output_dir, exist_ok=True)

sns.set_style("whitegrid")

# -----------------------------
# Identify variable types
# -----------------------------
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns

print("Numeric variables:", numeric_cols)
print("Categorical variables:", categorical_cols)

# -----------------------------
# 1 Missing Data Heatmap
# -----------------------------
plt.figure(figsize=(10,6))
sns.heatmap(df.isnull(), cbar=False)
plt.title("Missing Data Pattern")
plt.savefig(f"{output_dir}/missing_data_heatmap.png")
plt.close()

# -----------------------------
# 2 Numeric Variable Histograms
# -----------------------------
for col in numeric_cols:
    plt.figure()
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.savefig(f"{output_dir}/hist_{col}.png")
    plt.close()

# -----------------------------
# 3 Boxplots for Outliers
# -----------------------------
for col in numeric_cols:
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.savefig(f"{output_dir}/boxplot_{col}.png")
    plt.close()

# -----------------------------
# 4 Categorical Variable Plots
# -----------------------------
for col in categorical_cols:
    plt.figure(figsize=(6,4))
    sns.countplot(y=df[col], order=df[col].value_counts().index)
    plt.title(f"Distribution of {col}")
    plt.xlabel("Count")
    plt.ylabel(col)
    plt.savefig(f"{output_dir}/bar_{col}.png")
    plt.close()

# -----------------------------
# 5 Correlation Heatmap
# -----------------------------
if len(numeric_cols) > 1:
    plt.figure(figsize=(10,8))
    corr = df[numeric_cols].corr()

    sns.heatmap(corr,
                annot=True,
                cmap="coolwarm",
                fmt=".2f",
                square=True)

    plt.title("Correlation Matrix")
    plt.savefig(f"{output_dir}/correlation_heatmap.png")
    plt.close()

# -----------------------------
# 6 Pairplot
# -----------------------------
if len(numeric_cols) > 1:
    sns.pairplot(df[numeric_cols].dropna())
    plt.savefig(f"{output_dir}/pairplot.png")
    plt.close()

print("\nEDA visualization complete!")
print(f"Plots saved in: {output_dir}")