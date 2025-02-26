import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import keras_tuner as kt

# ---------------------------
# 1. LOAD & PREPARE THE DATA
# ---------------------------
df = pd.read_csv('AAPL.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.sort_index(inplace=True)

# Create a daily returns column (percentage change of Close)
df['Returns'] = df['Close'].pct_change()
df.dropna(inplace=True)  # Remove the first row with NaN

# Use multiple features; here we include: Open, High, Low, Close, Volume, YTD Gains, and Returns.
feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'YTD Gains', 'Returns']
data = df[feature_cols].copy()

# ---------------------------------
# 2. SCALE THE FEATURES
# ---------------------------------
# For simplicity, we use one scaler for all features.
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# -----------------------------------
# 3. CREATE A DIRECT MULTI-STEP DATASET
# -----------------------------------
# We'll use a 60-day lookback window to predict the next 5 days of returns (the last column).
time_step = 60     # Lookback period
future_days = 5    # Forecast horizon

def create_multi_step_dataset(dataset, time_step, future_days):
    X, y = [], []
    for i in range(len(dataset) - time_step - future_days + 1):
        X.append(dataset[i:i+time_step, :])
        # Predict the next 'future_days' of returns (last column)
        y.append(dataset[i+time_step:i+time_step+future_days, -1])
    return np.array(X), np.array(y)

X, y = create_multi_step_dataset(scaled_data, time_step, future_days)
print("X shape:", X.shape)  # (num_samples, 60, 7)
print("y shape:", y.shape)  # (num_samples, 5)

# -------------------------
# 4. TRAIN-TEST SPLIT (TIME-BASED)
# -------------------------
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# ---------------------------------------------------
# 5. DEFINE MODEL BUILDING FUNCTION WITH KERAS TUNER
# ---------------------------------------------------
def build_model(hp):
    model = Sequential()
    # First LSTM layer with tunable number of units and dropout
    model.add(LSTM(
        units=hp.Int('lstm_units_1', min_value=32, max_value=128, step=32),
        return_sequences=True,
        input_shape=(time_step, X_train.shape[2])
    ))
    model.add(Dropout(rate=hp.Float('dropout_rate_1', min_value=0.1, max_value=0.5, step=0.1)))
    
    # Second LSTM layer with tunable units and dropout
    model.add(LSTM(
        units=hp.Int('lstm_units_2', min_value=32, max_value=128, step=32),
        return_sequences=False
    ))
    model.add(Dropout(rate=hp.Float('dropout_rate_2', min_value=0.1, max_value=0.5, step=0.1)))
    
    # Output Dense layer with 'future_days' units to predict 5 future returns
    model.add(Dense(future_days))
    
    model.compile(optimizer='adam', loss='mse')
    return model

# -------------------------
# 6. SET UP THE KERAS TUNER
# -------------------------
tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=10,           # Try 10 different hyperparameter combinations
    executions_per_trial=1,  # Number of model evaluations per trial
    directory='hyperparam_tuning',
    project_name='lstm_tuning'
)

# -------------------------
# 7. RUN THE HYPERPARAMETER SEARCH
# -------------------------
tuner.search(X_train, y_train, epochs=10, validation_split=0.1, batch_size=32, verbose=1)

# Retrieve the best hyperparameters
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best Hyperparameters:")
print("LSTM Layer 1 Units:", best_hp.get('lstm_units_1'))
print("Dropout Layer 1 Rate:", best_hp.get('dropout_rate_1'))
print("LSTM Layer 2 Units:", best_hp.get('lstm_units_2'))
print("Dropout Layer 2 Rate:", best_hp.get('dropout_rate_2'))

# -------------------------
# 8. TRAIN THE FINAL MODEL WITH BEST HYPERPARAMETERS
# -------------------------
best_model = tuner.hypermodel.build(best_hp)
history = best_model.fit(
    X_train, y_train,
    epochs=50,
    validation_split=0.1,
    batch_size=32,
    verbose=1
)

# Evaluate the model on the test set
test_loss = best_model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)

# -------------------------
# 9. MAKE PREDICTIONS AND PLOT RESULTS
# -------------------------
y_pred = best_model.predict(X_test)

# -------------------------
# Plot 1: Scaled Future Returns Prediction
# -------------------------
plt.figure(figsize=(10, 6))
plt.plot(y_test[-1], marker='o', label='True Future Returns (scaled)')
plt.plot(y_pred[-1], marker='x', label='Predicted Future Returns (scaled)')
plt.title('Hyperparameter Tuned LSTM: Future Returns Prediction (Scaled)')
plt.xlabel('Days Ahead')
plt.ylabel('Scaled Return')
plt.legend()
plt.show()

# -------------------------
# Convert Predicted Scaled Returns to Actual Stock Price Forecast
# -------------------------
# Get the scaler parameters for the Returns column (index 6)
min_return = scaler.data_min_[6]
max_return = scaler.data_max_[6]

# Invert scaling for predicted returns (last test sample)
predicted_returns_scaled = y_pred[-1]  # shape: (5,)
predicted_returns_actual = predicted_returns_scaled * (max_return - min_return) + min_return

# For starting price, use the last day's Close from the last test sample's input window.
# The Close column is at index 3.
last_test_input = X_test[-1]
scaled_last_close = last_test_input[-1, 3]

# Get scaler parameters for Close column (index 3)
min_close = scaler.data_min_[3]
max_close = scaler.data_max_[3]
last_close_actual = scaled_last_close * (max_close - min_close) + min_close

# Reconstruct predicted Close prices using the actual predicted returns.
predicted_prices = [last_close_actual]
for ret in predicted_returns_actual:
    next_price = predicted_prices[-1] * (1 + ret)
    predicted_prices.append(next_price)
# Remove the seed price; forecasted prices are:
forecasted_prices = predicted_prices[1:]
print("Predicted Future Close Prices:", forecasted_prices)

# Generate forecast dates (assuming business days) starting from the last date in the dataset.
last_date = df.index[-1]
forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days, freq='B')

# -------------------------
# Plot 2: Historical Close and Forecasted Prices
# -------------------------
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Close'], label='Historical Close Price')
plt.plot(forecast_dates, forecasted_prices, 'ro--', label='Forecasted Close Price')
plt.title('Historical Close Price with Forecasted Future Prices')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
