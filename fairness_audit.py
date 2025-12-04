import pandas as pd

df = pd.read_csv("data/combined_civic.csv")

# Example: simple group fairness check by region
fairness_results = df.groupby('region')['label'].value_counts(normalize=True).unstack().fillna(0)
print(fairness_results)
fairness_results.to_csv("data/fairness_report.csv")
print("Fairness audit complete.")
