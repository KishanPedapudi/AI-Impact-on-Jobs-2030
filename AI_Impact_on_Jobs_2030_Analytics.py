"""
AI Impact on Jobs 2030 - Complete Analytics Pipeline
Runs directly on AI_Impact_on_Jobs_2030.csv in Google Colab
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("AI_Impact_on_Jobs_2030.csv")

print("="*50)
print("DATASET OVERVIEW")
print("="*50)
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

# -----------------------------
# Data Cleaning
# -----------------------------
df = df.drop_duplicates()

# Fill numeric nulls with median
numeric_cols = df.select_dtypes(include=["int64","float64"]).columns
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical nulls with mode
categorical_cols = df.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nClean Shape:", df.shape)

# -----------------------------
# Descriptive Statistics
# -----------------------------
print("\nSummary Statistics")
print(df.describe())

# -----------------------------
# Correlation Analysis
# -----------------------------
corr = df[numeric_cols].corr()

plt.figure(figsize=(12,8))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("heatmap.png")
plt.close()

# -----------------------------
# Salary Analysis
# -----------------------------
salary_by_job = (
    df.groupby("Job_Title")["Average_Salary_USD"]
    .mean()
    .sort_values(ascending=False)
)

print("\nTop 10 Highest Paying Jobs")
print(salary_by_job.head(10))

# -----------------------------
# AI Risk Analysis
# -----------------------------
risk_by_job = (
    df.groupby("Job_Title")["AI_Replacement_Risk"]
    .mean()
    .sort_values(ascending=False)
)

print("\nTop 10 Jobs at Highest AI Risk")
print(risk_by_job.head(10))

# -----------------------------
# Future Demand Analysis
# -----------------------------
future_jobs = (
    df.groupby("Job_Title")["Future_Demand_Score"]
    .mean()
    .sort_values(ascending=False)
)

print("\nTop 10 Future Demand Jobs")
print(future_jobs.head(10))

# -----------------------------
# Visualizations
# -----------------------------

plt.figure(figsize=(8,5))
sns.histplot(df["Average_Salary_USD"], bins=20)
plt.title("Salary Distribution")
plt.tight_layout()
plt.savefig("salary_distribution.png")
plt.close()

plt.figure(figsize=(8,5))
sns.scatterplot(
    data=df,
    x="Years_Experience",
    y="Average_Salary_USD"
)
plt.title("Experience vs Salary")
plt.tight_layout()
plt.savefig("experience_vs_salary.png")
plt.close()

plt.figure(figsize=(8,5))
sns.boxplot(
    data=df,
    x="Automation_Level",
    y="AI_Replacement_Risk"
)
plt.title("Automation Level vs AI Risk")
plt.tight_layout()
plt.savefig("automation_vs_risk.png")
plt.close()

# -----------------------------
# Export Clean Dataset
# -----------------------------
df.to_csv("AI_Impact_on_Jobs_2030_Clean.csv", index=False)

print("\nFiles Generated:")
print("- heatmap.png")
print("- salary_distribution.png")
print("- experience_vs_salary.png")
print("- automation_vs_risk.png")
print("- AI_Impact_on_Jobs_2030_Clean.csv")
