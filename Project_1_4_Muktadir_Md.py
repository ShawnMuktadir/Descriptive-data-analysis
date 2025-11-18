# ---------------------------------------------------------
#  Investigate 3 additional aspects of your choice
# ---------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
#  Load Data
# -------------------------------
file = r"data/Happiness.csv"
df = pd.read_csv(file, sep=';', engine='python')

print(" Data loaded successfully!")
print(f"Initial shape: {df.shape}")

# -------------------------------
#  Data Cleaning
# -------------------------------
# Replace commas with dots and strip spaces
df = df.map(lambda x: str(x).replace(',', '.').strip() if isinstance(x, str) else x)

# Convert to numeric where possible
for col in df.columns:
    try:
        df[col] = pd.to_numeric(df[col])
    except Exception:
        pass  # Skip non-numeric columns

# Handle missing values
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df.groupby('Country name')[col].transform(lambda x: x.fillna(x.mean()))
        df[col] = df[col].fillna(df[col].mean())
    else:
        mode_val = df[col].mode()
        if not mode_val.empty:
            df[col] = df[col].fillna(mode_val[0])

print(" Missing values handled.\n")

# -------------------------------
#  Analyze Key Factors
# -------------------------------
factors = ['Log GDP per capita', 'Social support', 'Perceptions of corruption']

for factor in factors:
    # Aggregate by country: mean value of the factor
    country_avg = df.groupby('Country name', as_index=False)[factor].mean()

    # Top and bottom 10 countries
    top_10 = country_avg.nlargest(10, factor)
    bottom_10 = country_avg.nsmallest(10, factor)

    print(f"\n Top 10 Countries by {factor}:")
    print(top_10)

    print(f"\n Bottom 10 Countries by {factor}:")
    print(bottom_10)

    # --- Plot Top 10 ---
    plt.figure(figsize=(10, 5))
    sns.barplot(data=top_10, x=factor, y='Country name', hue='Country name', palette='Greens_r', legend=False)
    plt.title(f"Top 10 Countries by {factor}")
    plt.xlabel(f"{factor} (Average)")
    plt.ylabel("Country")
    plt.tight_layout()
    plt.show()

    # --- Plot Bottom 10 ---
    plt.figure(figsize=(10, 5))
    sns.barplot(data=bottom_10, x=factor, y='Country name', hue='Country name', palette='Reds', legend=False)
    plt.title(f"Bottom 10 Countries by {factor}")
    plt.xlabel(f"{factor} (Average)")
    plt.ylabel("Country")
    plt.tight_layout()
    plt.show()

    # -------------------------------
    #  Save Results to CSV
    # -------------------------------
    clean_name = factor.lower().replace(" ", "_")
    top_10.to_csv(f"data/appendix_top10_{clean_name}.csv", index=False)
    bottom_10.to_csv(f"data/appendix_bottom10_{clean_name}.csv", index=False)
    print(f" Saved: appendix_top10_{clean_name}.csv and appendix_bottom10_{clean_name}.csv")

print("\n All tables and plots generated successfully!")
