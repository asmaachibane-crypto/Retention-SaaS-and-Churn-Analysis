import pandas as pd

REQUIRED_COLUMNS = [
    "user_id",
    "signup_date",
    "last_activity_date",
    "subscription_type",
    "monthly_fee",
    "usage_frequency",
    "region"
]

OPTIONAL_COLUMNS = [
    "churn_reason"
]


def validate_structure(df):
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    return True


def enforce_types(df):

    # Convert dates
    df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")
    df["last_activity_date"] = pd.to_datetime(df["last_activity_date"], errors="coerce")

    # Convert monthly fee to numeric
    df["monthly_fee"] = pd.to_numeric(df["monthly_fee"], errors="coerce")

    # Convert usage_frequency categories to numbers
    frequency_map = {
        "Daily": 30,
        "Weekly": 4,
        "Monthly": 1,
        "Rare": 0.5
    }

    df["usage_frequency"] = df["usage_frequency"].map(frequency_map)

    return df


def clean_data(df):

    # Remove duplicate users
    df = df.drop_duplicates(subset=["user_id"])

    # Remove rows with invalid dates
    df = df.dropna(subset=["signup_date", "last_activity_date"])

    # Fill missing categorical values
    df["region"] = df["region"].fillna("Unknown")

    # Fill missing numeric values
    df["usage_frequency"] = df["usage_frequency"].fillna(0)
    df["monthly_fee"] = df["monthly_fee"].fillna(df["monthly_fee"].median())

    return df


def create_features(df, inactivity_threshold=30):

    reference_date = df["last_activity_date"].max()

    # Days inactive
    df["days_inactive"] = (reference_date - df["last_activity_date"]).dt.days

    # Churn definition
    df["churn"] = (df["days_inactive"] > inactivity_threshold).astype(int)

    # Tenure
    df["tenure_days"] = (df["last_activity_date"] - df["signup_date"]).dt.days
    df["tenure_days"] = df["tenure_days"].clip(lower=1)

    # Activity rate
    df["activity_rate"] = df["usage_frequency"] / df["tenure_days"]
    df["activity_rate"] = df["activity_rate"].fillna(0)

    return df


def preprocess_pipeline(df):

    validate_structure(df)

    df = enforce_types(df)

    df = clean_data(df)

    df = create_features(df)

    return df