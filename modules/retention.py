import pandas as pd


def create_cohort_data(df):
    df["signup_month"] = df["signup_date"].dt.to_period("M")
    df["activity_month"] = df["last_activity_date"].dt.to_period("M")

    df["cohort_age"] = (
        df["activity_month"] - df["signup_month"]
    ).apply(lambda x: x.n)

    return df


def compute_retention_matrix(df):
    cohort_data = (
        df.groupby(["signup_month", "cohort_age"])["user_id"]
        .nunique()
        .reset_index()
    )

    cohort_sizes = df.groupby("signup_month")["user_id"].nunique()

    cohort_data["retention_rate"] = cohort_data.apply(
        lambda row: row["user_id"] / cohort_sizes[row["signup_month"]],
        axis=1
    )

    retention_matrix = cohort_data.pivot(
        index="signup_month",
        columns="cohort_age",
        values="retention_rate"
    )

    return retention_matrix
