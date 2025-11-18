# ---------------------------------------------------------
#  Comparison of Hong Kong, China, and Germany for Paul
# ---------------------------------------------------------

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------
#  Load Data
# ---------------------------------------------------------
file = r"data/Happiness.csv"
df = pd.read_csv(file, sep=';', engine='python')

print(" Data loaded successfully!")
print(f"Shape: {df.shape}\n")

# ---------------------------------------------------------
#  Convert Number Formats and Clean
# ---------------------------------------------------------
# Use .map instead of deprecated .applymap
df = df.map(lambda x: str(x).replace(',', '.').strip() if isinstance(x, str) else x)

# Convert to numeric where possible
for col in df.columns:
    try:
        df[col] = pd.to_numeric(df[col])
    except Exception:
        pass  # leave non-numeric columns as-is

# ---------------------------------------------------------
# Handle Missing Values
# ---------------------------------------------------------
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df.groupby('Country name')[col].transform(lambda x: x.fillna(x.mean()))
        df[col] = df[col].fillna(df[col].mean())  # no inplace=True to avoid warning
    else:
        mode_val = df[col].mode()
        if not mode_val.empty:
            df[col] = df[col].fillna(mode_val[0])

print(" Missing values handled.\n")

# ---------------------------------------------------------
# Correlation Analysis: Find Top Happiness Factors
# ---------------------------------------------------------
corr = df.corr(numeric_only=True)["Life Ladder"].sort_values(ascending=False)
top_factors = corr.drop('Life Ladder').head(5).index.tolist()

print(" Top 5 most happiness-related factors:")
print(top_factors, "\n")

# ---------------------------------------------------------
# Compare Selected Countries & Show Exact Values
# ---------------------------------------------------------
countries = ["Hong Kong S.A.R. of China", "China", "Germany"]
compare_df = df[df["Country name"].isin(countries)]

compare_means = compare_df.groupby("Country name")[top_factors].mean().reset_index()

print(" Average values of Top 5 happiness-related factors (2008–2018):\n")
print(compare_means.to_string(index=False))

# Prepare for visualization
compare_melted = compare_means.melt(
    id_vars="Country name",
    var_name="Factor",
    value_name="Average Score"
)

# ---------------------------------------------------------
# Visualization: Compare on Happiness-Related Factors
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
sns.barplot(
    data=compare_melted,
    x="Factor",
    y="Average Score",
    hue="Country name",
    palette="Set2"
)
plt.title("Comparison of Top Happiness-Related Factors\n(Hong Kong vs China vs Germany)")
plt.xlabel("Factor")
plt.ylabel("Average Score (2008–2018)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('data/country_comparison.png', dpi=300)
plt.show()

# ---------------------------------------------------------
#  Weighted Happiness Potential Score
# ---------------------------------------------------------
weights = corr[top_factors]
weighted_scores = compare_means.set_index("Country name")[top_factors].mul(weights, axis=1).sum(axis=1)
weighted_scores = weighted_scores.sort_values(ascending=False)

print("\n Weighted Happiness Potential Score (Correlation-Weighted):")
print(weighted_scores)

# Visualization — fix seaborn palette warning by using hue
plt.figure(figsize=(10, 6))
sns.barplot(
    data=weighted_scores.reset_index(),
    x=0,
    y="Country name",
    hue="Country name",
    palette="coolwarm",
    legend=False
)
plt.title("Weighted Happiness Potential Score (Correlation-Weighted)")
plt.xlabel("Weighted Score")
plt.ylabel("Country")
plt.tight_layout()
plt.savefig('data/country_correlation.png', dpi=300)
plt.show()
