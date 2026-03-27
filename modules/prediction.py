import pandas as pd


def monthly_active_users(df):
    #Calcule le nombre d'utilisateurs uniques par mois
    df_copy = df.copy()
    df_copy["month"] = df_copy["month"].astype(str)
    monthly_users = df_copy.groupby("month")["user_id"].nunique()
    return monthly_users


def project_users(monthly_users, months_ahead=6):
    #Projette les utilisateurs pour les mois à venir en utilisant 
    # le taux de croissance moyen

    growth_rate = monthly_users.pct_change().mean()
    last_value = monthly_users.iloc[-1]

    projections = []

    for i in range(months_ahead):
        last_value = last_value * (1 + growth_rate)
        projections.append(last_value)

    return projections


def scenario_projection(monthly_users, months_ahead=6):
    #Projette les utilisateurs selon trois scénarios : pessimiste, réaliste et optimiste
    growth_rates = monthly_users.pct_change().fillna(0)
    avg_growth = growth_rates.mean()

    scenarios = {
        "pessimistic": avg_growth - 0.05,
        "realistic": avg_growth,
        "optimistic": avg_growth + 0.05
    }

    results = {}
    for scenario, rate in scenarios.items():
        last_value = monthly_users.iloc[-1]
        projections = [(last_value * ((1 + rate) ** (i+1))) for i in range(months_ahead)]
        results[scenario] = projections

    return results