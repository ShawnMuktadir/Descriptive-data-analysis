# ------------------------------- Total score -------------------------------
# Which countries have the highest/lowest overall score and are therefore
# particularly suitable as a new home or are more likely to be excluded?
# Which factors have the strongest associations with the happiness score?
# ---------------------------------------------------------------------------


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
#  Load Data
# -------------------------------
file = r"data/Happiness.csv"
df = pd.read_csv(file, sep=';', engine='python')

print("Data loaded successfully!")
print(f"Initial shape: {df.shape}")

rows, cols = df.shape
print(f"Number of rows: {rows}")
print(f"Number of columns: {cols}")

# Missing values per column
missing_values = df.isnull().sum()
print("\nMissing values per column:")
print(missing_values)

# -------------------------------
# Convert Number Formats
# -------------------------------
# Replace commas with dots in all cells and strip spaces
df = df.map(lambda x: str(x).replace(',', '.').strip() if isinstance(x, str) else x)

# Convert numeric columns safely
for col in df.columns:
    try:
        df[col] = pd.to_numeric(df[col])
    except ValueError:
        pass  # Keep non-numeric columns unchanged

# -------------------------------
# Handle Missing Values
# -------------------------------
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df.groupby('Country name')[col].transform(lambda x: x.fillna(x.mean()))
        df[col] = df[col].fillna(df[col].mean())
    else:
        mode_val = df[col].mode()
        if not mode_val.empty:
            df[col] = df[col].fillna(mode_val[0])

print("Missing values handled.")
print(df.dtypes.head(10))  # Check numeric conversion

# -------------------------------
# Descriptive Statistics
# -------------------------------
print("\nSummary Statistics:")
print(df.describe().T.head(10))

# -------------------------------
# Average Happiness per Country
# -------------------------------
avg_happiness = df.groupby("Country name")["Life Ladder"].mean().sort_values(ascending=False)
top_10 = avg_happiness.head(10)
bottom_10 = avg_happiness.tail(10)

print("\n Top 10 Happiest Countries:")
print(top_10)

print("\n Bottom 10 Least Happy Countries:")
print(bottom_10)

# -------------------------------
# Visualize Happiness Rankings
# -------------------------------
plt.figure(figsize=(10, 5))
sns.barplot(x=top_10.values, y=top_10.index, hue=top_10.index, palette="Greens_r", legend=False)
plt.title("Top 10 Happiest Countries (Average 2008–2018)")
plt.xlabel("Average Life Ladder (Happiness Score)")
plt.ylabel("Country")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(x=bottom_10.values, y=bottom_10.index, hue=bottom_10.index, palette="Reds", legend=False)
plt.title("Bottom 10 Least Happy Countries (Average 2008–2018)")
plt.xlabel("Average Life Ladder (Happiness Score)")
plt.ylabel("Country")
plt.tight_layout()
plt.show()

# -------------------------------
# Correlation with Happiness
# -------------------------------
corr = df.corr(numeric_only=True)["Life Ladder"].sort_values(ascending=False)
print("\n📈 Correlation with Life Ladder:")
print(corr)

plt.figure(figsize=(8, 5))
corr.drop("Life Ladder").head(8).plot(kind='barh', color='skyblue')
plt.title("Top Factors Associated with Happiness")
plt.xlabel("Correlation Coefficient")
plt.tight_layout()
plt.show()
