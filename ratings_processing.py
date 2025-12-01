import pandas as pd
from datetime import datetime

# ===== 1. Rating Text Normalization =====

def normalize_rating(raw):
    """
    Normalize raw rating text to canonical form: buy / hold / sell.
    """
    if not isinstance(raw, str):
        return None
    r = raw.strip().lower()
    if not r:  # Handle empty strings
        return None

    # --- buy family ---
    if any(k in r for k in [
        "strong buy", "outperform", "overweight", "accumulate", "add"
    ]):
        return "buy"
    if "buy" in r:
        return "buy"

    # --- hold family ---
    if any(k in r for k in [
        "hold", "neutral", "market perform", "market-perform",
        "equal weight", "equal-weight", "perform"
    ]):
        return "hold"

    # --- sell family ---
    if any(k in r for k in [
        "strong sell", "underperform", "underweight", "sell", "reduce"
    ]):
        return "sell"

    return None


def rating_to_numeric(canonical):
    """
    Map canonical rating to numeric value.
    buy = 1, hold = 0, sell = -1
    """
    mapping = {"buy": 1, "hold": 0, "sell": -1}
    if canonical is None:
        return None
    return mapping.get(canonical, None)


# ===== 2. Main Processing Function =====

def process_ratings(
    input_csv: str,
    output_csv: str
):
    """
    Read raw broker ratings CSV -> Generate features -> Output to new CSV.
    """

    df = pd.read_csv(input_csv)

    # ---- 2.1 Column Mapping (Unified to internal column names) ----
    column_mapping = {
        "GradeDate": "date",
        "Firm": "broker",
        "ToGrade": "to_grade",
        "FromGrade": "from_grade",
        "Action": "action",
        "priceTargetAction": "price_target_action",
        "currentPriceTarget": "new_target",
        "priorPriceTarget": "old_target"
    }
    
    # Rename only existing columns
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

    # ---- 2.2 Standard Time Format ----
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # ---- 2.3 Normalize Rating Text ----
    df["to_canonical"] = df["to_grade"].apply(normalize_rating)
    # FromGrade may be empty, needs handling
    df["from_canonical"] = df["from_grade"].fillna("").apply(normalize_rating)

    # ---- 2.4 Numerical Rating ----
    df["to_numeric"] = df["to_canonical"].apply(rating_to_numeric)
    df["from_numeric"] = df["from_canonical"].apply(rating_to_numeric)

    # ---- 2.5 Rating Change (Upgrade / Downgrade) ----
    # If from_numeric is empty, rating_delta will also be NaN, needs handling
    df["rating_delta"] = df["to_numeric"] - df["from_numeric"]
    df["rating_delta"] = df["rating_delta"].fillna(0)  # If no old rating, set to 0 (no change)

    df["upgrade_flag"] = (df["rating_delta"] > 0).astype(int)
    df["downgrade_flag"] = (df["rating_delta"] < 0).astype(int)

    # ---- 2.6 Price Target Related Features ----
    # First ensure it is numeric, handle 0.0 and null values
    df["new_target"] = pd.to_numeric(df["new_target"], errors="coerce")
    df["old_target"] = pd.to_numeric(df["old_target"], errors="coerce")
    
    # Treat 0.0 as missing value
    df["new_target"] = df["new_target"].replace(0.0, pd.NA)
    df["old_target"] = df["old_target"].replace(0.0, pd.NA)

    df["target_delta"] = df["new_target"] - df["old_target"]

    # Percentage change: (New - Old) / Old, avoid division by 0 or missing values
    # Calculate percentage only when old_target exists and is not 0
    mask_valid = (df["old_target"].notna()) & (df["old_target"] != 0)
    df["target_pct_change"] = pd.NA
    df.loc[mask_valid, "target_pct_change"] = (
        df.loc[mask_valid, "target_delta"] / df.loc[mask_valid, "old_target"]
    )

    # Direction: Raise / Cut / Unchanged
    df["tp_revision_direction"] = 0
    df.loc[df["target_delta"] > 0, "tp_revision_direction"] = 1
    df.loc[df["target_delta"] < 0, "tp_revision_direction"] = -1

    # ---- 2.7 Categorize Action ----
    def categorize_action(row):
        # Prioritize priceTargetAction (more accurate)
        pt_action = str(row.get("price_target_action", "")).lower()
        action = str(row.get("action", "")).lower()
        delta = row["rating_delta"]

        # Rating Change
        if delta > 0:
            return "upgrade"
        if delta < 0:
            return "downgrade"

        # Rating unchanged, check priceTargetAction
        if "raise" in pt_action:
            return "tp_raise"
        if "lower" in pt_action:
            return "tp_cut"
        if "maintain" in pt_action:
            return "reiterate"
        if "announce" in pt_action:
            return "announce"
        if "adjust" in pt_action:
            return "adjust"

        # fallback: Check price target change direction
        if row["tp_revision_direction"] > 0:
            return "tp_raise"
        if row["tp_revision_direction"] < 0:
            return "tp_cut"
        
        # If action is init, it means initial rating
        if "init" in action:
            return "init"
        
        return "other"

    df["action_category"] = df.apply(categorize_action, axis=1)

    # ---- 2.8 Sort by Date for Time Series Analysis ----
    df = df.sort_values("date")

    # ---- 2.9 Save Results ----
    df.to_csv(output_csv, index=False)
    print(f"Done. Processed {len(df)} ratings. Saved to: {output_csv}")
    return df


if __name__ == "__main__":
    input_file = "TSLA_ratings_yfinance_2020_2025.csv"
    process_ratings(input_file, "ratings_output.csv")
