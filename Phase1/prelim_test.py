import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load Tesla data
df = yf.download("TSLA", start="2015-01-01", end="2020-01-01")
df = df[['Open', 'Close', 'Volume']]

# Normalize data
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled, columns=df.columns)

# Build supervised dataset
def create_sequences(data, seq_len, n_future):
    X, y = [], []
    for i in range(len(data) - seq_len-n_future):
        X.append(data[i:i+seq_len]) #past 30 days
        y.append(data[i+seq_len:i+seq_len+n_future, 1])  # next 7 days
    return np.array(X), np.array(y)

#parameters
seq_len = 30
n_future = 7

X, y = create_sequences(scaled_df.values, seq_len, n_future)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

·
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    #Dropout(0.2),
    LSTM(64),
    #Dropout(0.2),
    Dense(n_future)
])

model.compile(optimizer='adam', loss='mse')
model.summary()

model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)


y_pred = model.predict(X_test)

# Inverse transform for readability
def inverse_transform_close(pred):
    zeros = np.zeros((pred.shape[0], 2))  # dummy Open & Volume
    merged = np.concatenate([zeros, pred], axis=1)
    return scaler.inverse_transform(merged)[:, 2]

y_test_inv = inverse_transform_close(y_test.flatten().reshape(-1, 1))
y_pred_inv = inverse_transform_close(y_pred.flatten().reshape(-1, 1))

plt.figure(figsize=(10,5))
plt.plot(y_test_inv, label="Actual Close Price")
plt.plot(y_pred_inv, label="Predicted (next 7 days))")
plt.xlabel("Days")
plt.ylabel("Price (USD)")
plt.title("Actual vs Predicted (next 7 days)")
plt.legend()
plt.show()