import pandas as pd
import streamlit as st

# Charger un fichier CSV
def load_data(file):

    try:
        df = pd.read_csv(file)
        return df

    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


# Vérifier les colonnes nécessaires
def validate_dataset(df):

    required_columns = [
        "customer_id",
        "subscription_date",
        "last_login",
        "monthly_revenue",
        "logins_last_30_days",
        "feature_usage_score",
        "support_tickets",
        "churned"
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        st.error(f"Missing columns: {missing}")
        return False

    return True


# Convertir les dates
def convert_dates(df):

    df["subscription_date"] = pd.to_datetime(df["subscription_date"])
    df["last_login"] = pd.to_datetime(df["last_login"])

    return df