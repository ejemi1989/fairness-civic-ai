import pandas as pd

df = pd.read_csv("data/fairness_report.csv")

with open("civic_fairness_report.md", "w") as f:
    f.write("# Civic AI Fairness Report\n\n")
    f.write("## Label Distribution by Region\n\n")
    f.write(df.to_markdown())
print("Report generated: civic_fairness_report.md")
