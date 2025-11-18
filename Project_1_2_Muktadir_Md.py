# ---------------------------------------------------------
# Gini Index: Determine the Gini index for the last year
# ---------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Load Data
# -------------------------------
file = r"data/Happiness.csv"
df = pd.read_csv(file, sep=';', engine='python')

print(" Data loaded successfully!")
print(f"Initial shape: {df.shape}\n")

# -------------------------------
# Data Cleaning
# -------------------------------
# Replace commas with dots and strip spaces (apply to all elements safely)
df = df.map(lambda x: str(x).replace(',', '.').strip() if isinstance(x, str) else x)

# Convert numeric columns safely
for col in df.columns:
    try:
        df[col] = pd.to_numeric(df[col])
    except ValueError:
        pass  # keep non-numeric columns unchanged

# Fill missing numeric values by country mean, then global mean
for col in df.select_dtypes(include='number').columns:
    country_means = df.groupby('Country name')[col].transform('mean')
    df[col] = df[col].fillna(country_means)
    df[col] = df[col].fillna(df[col].mean())

# Fill missing categorical values with mode
for col in df.select_dtypes(include='object').columns:
    mode_val = df[col].mode()
    if not mode_val.empty:
        df[col] = df[col].fillna(mode_val[0])

print(" Missing values handled.\n")

# -------------------------------
# Focus on GINI-related columns
# -------------------------------
gini_cols = [
    "GINI index (World Bank estimate)",
    "GINI index (World Bank estimate), average 2000-16",
    "gini of household income reported in Gallup, by wp5-year"
]

print("Missing values in GINI columns:")
print(df[gini_cols].isnull().sum(), "\n")

# -------------------------------
# Correlation with Life Ladder
# -------------------------------
numeric_cols = gini_cols + ['Life Ladder']
df_corr = df[numeric_cols].dropna()
corr_matrix = df_corr.corr()

print("Correlation of GINI columns with Life Ladder:")
print(corr_matrix['Life Ladder'].sort_values(ascending=False), "\n")

# -------------------------------
# Heatmap Visualization
# -------------------------------
plt.figure(figsize=(10, 6))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='coolwarm',
    fmt=".2f",
    linewidths=0.5,
    annot_kws={"size": 10}
)
plt.xticks(rotation=10, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.title("Correlation between GINI indexes and Life Ladder", fontsize=12)
plt.tight_layout()
plt.savefig('data/gini_correlation.png')
plt.show()

# -------------------------------
# Extract GINI for the last year
# -------------------------------
last_year = df['Year'].max()
df_last_year = df[df['Year'] == last_year].copy()

# Use Gallup GINI as primary, fallback to World Bank GINI
df_last_year['GINI_final'] = df_last_year[
    'gini of household income reported in Gallup, by wp5-year'
].fillna(df_last_year['GINI index (World Bank estimate)'])

# Sort descending by inequality
gini_last_year = df_last_year[['Country name', 'GINI_final']].sort_values(by='GINI_final', ascending=False)

# -------------------------------
# Print all countries with GINI for last year
# -------------------------------
pd.set_option('display.max_rows', None)
print(f" GINI index for all countries in {last_year}:")
print(gini_last_year)

# Save the GINI index table to CSV for report use
gini_last_year.to_csv("data\gini_index_2018.csv", index=False)
print(" Saved: gini_index_2018.csv")

