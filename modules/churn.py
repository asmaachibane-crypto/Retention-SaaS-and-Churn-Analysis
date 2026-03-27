import pandas as pd
def compute_global_churn(df):
    #Retourne le pourcentage global de churn
    total_users = len(df)
    total_churned = df["churn"].sum()
    return total_churned / total_users

def churn_by_subscription(df):
    #Churn moyen par type d'abonnement
    return df.groupby("subscription_type")["churn"].agg(["mean", "count"]).sort_values("mean", ascending=False)

def churn_by_region(df):
    #Churn moyen par région
    return df.groupby("region")["churn"].agg(["mean", "count"]).sort_values("mean", ascending=False)

def churn_reason_distribution(df):
    #Distribution des raisons de churn
    if "churn_reason" not in df.columns:
        return None
    churned = df[df["churn"] == 1]
    return churned["churn_reason"].value_counts(normalize=True)