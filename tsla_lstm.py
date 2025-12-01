import numpy as np
import pandas as pd
from math import floor

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, Flatten, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

# ======================
# 1. Load the data from the CSV file
# ======================

df = pd.read_csv("TSLA_2020_2024_total.csv")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# all numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns

# fill 0 for rating/news columns that are empty (no event happened → signal is 0)
df[numeric_cols] = df[numeric_cols].fillna(0)

# ======================
# 2. Aggregate by trading day (multiple rows per day → one row per day)
# ======================

daily = (
    df.groupby("date")[numeric_cols]
      .mean()
      .reset_index()
      .sort_values("date")
)

# ======================
# 3. Construct log return as the prediction target
# ======================

daily["log_return_1d"] = np.log(daily["Close"]).diff().shift(-1)
daily = daily.dropna(subset=["log_return_1d"]).reset_index(drop=True)

# threshold = 0.001
# r = daily["log_return_1d"]
# daily["direction_label"] = 0
# daily.loc[r >  threshold, "direction_label"] = 1
# daily.loc[r < -threshold, "direction_label"] = -1


# ======================
# 4. Time split (70% train, 15% val, 15% test)
# ======================

n = len(daily)
n_train = floor(n * 0.7)
n_val   = floor(n * 0.15)

train_df = daily.iloc[:n_train]
val_df   = daily.iloc[n_train:n_train+n_val]
test_df  = daily.iloc[n_train+n_val:]

# ======================
# 5. Three sets of features
# ======================

baseline_cols = [
    "Close","High","Low","Open","Volume",
    "MA5","MA10","MA20","MA50",
    "EMA20","EMA50",
    "MACD","MACD_Signal","MACD_Hist",
    "ROC","RSI14",
    "Stoch_%K","Stoch_%D",
    "ATR14",
    "BB_Middle","BB_Upper","BB_Lower",
    "HistVol20","OBV",
    "Vol_MA5","Vol_MA10","Vol_MA20",
    "DayOfWeek","Month"
]

improved_cols = baseline_cols + ["news_score", "rating_score"]

full_cols = [
    "Close","High","Low","Open","Volume",
    "MA5","MA10","MA20","MA50",
    "EMA20","EMA50",
    "MACD","MACD_Signal","MACD_Hist",
    "ROC","RSI14",
    "Stoch_%K","Stoch_%D",
    "ATR14",
    "BB_Middle","BB_Upper","BB_Lower",
    "HistVol20","OBV",
    "Vol_MA5","Vol_MA10","Vol_MA20",
    "DayOfWeek","Month",

    # social media
    "total_tweet_count",
    "avg_sentiment_positive","avg_sentiment_negative","avg_sentiment_neutral",
    "avg_sentiment_polarity",
    "max_sentiment_polarity","min_sentiment_polarity","std_sentiment_polarity",
    "pre_close_tweet_count","tesla_tweet_count","total_tesla_keywords",
    "total_likes","total_retweets","total_replies","total_views",
    "pct_tesla_tweets","pct_pre_close_tweets","weighted_sentiment",

    # news + rating
    "news_score","rating_score",
]


# ======================
# 6. Run the model
# ======================

LOOKBACK = 30

def run_model(feature_cols, model_name):

    print("\n======================")
    print(model_name)
    print("======================")

    # ---------- 6.1 Standardization (fit only on train) ----------
    scaler = StandardScaler()

    train_scaled = train_df.copy()
    val_scaled   = val_df.copy()
    test_scaled  = test_df.copy()

    train_scaled[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_scaled[feature_cols]   = scaler.transform(val_df[feature_cols])
    test_scaled[feature_cols]  = scaler.transform(test_df[feature_cols])

    # ---------- 6.2 Construct LSTM sequences ----------
    def build_sequences(df):
        X_list, y_list = [], []
        data = df[feature_cols].values
        target = df["log_return_1d"].values
        #target = df["direction_label"].values
        for i in range(len(df) - LOOKBACK):
            X_list.append(data[i:i+LOOKBACK])
            y_list.append(target[i+LOOKBACK])
        return np.array(X_list), np.array(y_list)

    X_train, y_train = build_sequences(train_scaled)
    X_val,   y_val   = build_sequences(val_scaled)
    X_test,  y_test  = build_sequences(test_scaled)

    print("Train:", X_train.shape, y_train.shape)
    print("Val:  ", X_val.shape, y_val.shape)
    print("Test: ", X_test.shape, y_test.shape)

    # ---------- 6.3 Define model ----------
    model = Sequential([
        Flatten(),  # Flatten 3D input (batch, timesteps, features) to 2D (batch, timesteps*features)
        Dense(512),
        LeakyReLU(alpha=0.01),
        Dense(64),
        LeakyReLU(alpha=0.01),
        Dense(8),
        LeakyReLU(alpha=0.01),
        Dense(1)
    ])
        
    model.compile(optimizer="adam", loss="mse")

    # ---------- 6.4 Train the model ----------
    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[es],
        verbose=1
    )

    # ---------- 6.5 Predict the model ----------
    y_pred = model.predict(X_test, verbose=0).ravel()

    # ---------- 6.6 Evaluate the model ----------
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    direction_acc = np.mean(np.sign(y_pred) == np.sign(y_test))

    print(f"{model_name} results:")
    print(f"RMSE = {rmse:.6f}")
    print(f"MAE  = {mae:.6f}")
    print(f"Direction accuracy = {direction_acc:.4f}")

    
    # ---------- 6.7 Plot results ----------
    # Get test dates for plotting
    test_dates = test_df.iloc[LOOKBACK:]["date"].values
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{model_name} - Model Performance', fontsize=14, fontweight='bold')
    
    # 1. Training history (loss curves)
    ax1 = axes[0, 0]
    ax1.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=10)
    ax1.set_ylabel('Loss (MSE)', fontsize=10)
    ax1.set_title('Training History\n(Lower is better)', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. Predicted vs Actual (Time Series)
    ax2 = axes[0, 1]
    ax2.plot(test_dates, y_test, label='Actual', linewidth=2, alpha=0.7)
    ax2.plot(test_dates, y_pred, label='Predicted', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Date', fontsize=10)
    ax2.set_ylabel('Log Return (1-day)', fontsize=10)
    ax2.set_title('Predicted vs Actual (Time Series)\n(Tracking performance over time)', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 3. Predicted vs Actual (Scatter Plot)
    ax3 = axes[1, 0]
    ax3.scatter(y_test, y_pred, alpha=0.5, s=15)
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax3.set_xlabel('Actual Log Return', fontsize=10)
    ax3.set_ylabel('Predicted Log Return', fontsize=10)
    ax3.set_title('Predicted vs Actual (Scatter)\n(Points on red line are perfect)', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4. Residuals plot
    ax4 = axes[1, 1]
    residuals = y_test - y_pred
    ax4.scatter(y_pred, residuals, alpha=0.5, s=15)
    ax4.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax4.set_xlabel('Predicted Log Return', fontsize=10)
    ax4.set_ylabel('Residuals (Actual - Predicted)', fontsize=10)
    ax4.set_title('Residuals Plot\n(Random spread around 0 is good)', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()

    print("\n[Plot Explanations]")
    print("1. Training History: Shows how model error decreases during training. Gap between lines indicates overfitting.")
    print("2. Time Series: Compares actual vs predicted stock returns over time. Look for pattern matching.")
    print("3. Scatter Plot: X-axis is Actual, Y-axis is Predicted. Points closer to the red diagonal line are better.")
    print("4. Residuals Plot: Shows prediction errors. Ideally should be random noise centered at 0 (no systematic bias).")
    

    plt.show()
    
    
    return model


# ======================
# 7. Run the three models
# ======================

baseline_model = run_model(baseline_cols, "Baseline(technical indicators)")
improved_model = run_model(improved_cols, "Improved(technical + news + rating)")
full_model     = run_model(full_cols, "Full(technical + social + news + rating)")
