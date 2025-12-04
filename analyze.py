from data_api import fetch_chicago, fetch_nyc
from preprocessing import preprocess
from fairness_metrics import compute_parity
import pandas as pd
import mlflow

if __name__ == "__main__":
    # Pull APIs
    chicago = fetch_chicago()
    nyc = fetch_nyc()

    # Combine and preprocess
    df = pd.concat([chicago, nyc], ignore_index=True)
    df = preprocess(df, categorical_cols=[1,2])  # adjust based on columns

    # Track experiments
    mlflow.start_run(run_name="civic_fairness_api")

    # Compute fairness (example)
    parity = compute_parity(df, target_col=1, group_col=2)
    print("Demographic Parity Differences:", parity)

    mlflow.end_run()
