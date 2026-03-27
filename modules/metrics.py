import pandas as pd


def calculate_mrr(df):
    #Calcule le revenu mensuel récurrent en ne considérant 
    # que les utilisateurs actifs
    active_users = df[df["churn"] == 0]
    return active_users["monthly_fee"].sum()


def calculate_arpu(df):
    #Calcule le revenu moyen par utilisateur
    return df["monthly_fee"].mean()


def calculate_ltv(df):
    #Calcule la valeur à vie d'un utilisateur
    avg_revenue = df["monthly_fee"].mean()
    avg_tenure = df["tenure_days"].mean() / 30
    return avg_revenue * avg_tenure
