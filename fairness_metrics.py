def compute_parity(df, target_col, group_col):
    groups = df[group_col].unique()
    overall_rate = df[target_col].mean()
    parity_diffs = {g: df[df[group_col]==g][target_col].mean() - overall_rate for g in groups}
    return parity_diffs
