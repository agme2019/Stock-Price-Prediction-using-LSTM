import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 1. Load and Prepare the Data
data = pd.read_csv('AAPL.csv')
print(data.head())
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data.sort_index(inplace=True)  # Ensure chronological order

# Extract the 'Close' column
close_data = data[['Close']]

# 2. Scale the Data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_data)

# 3. Create Sequences
def create_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(len(dataset) - time_step):
        X.append(dataset[i : i + time_step, 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X_all, y_all = create_dataset(scaled_data, time_step=time_step)

# We also need corresponding dates for the labels.
# The label for i-th sequence is at index (i + time_step).
all_dates = close_data.index
label_dates = all_dates[time_step:]  # Each label corresponds to this shifted index

# 4. Train-Test Split
train_size = int(len(X_all) * 0.8)
X_train, X_test = X_all[:train_size], X_all[train_size:]
y_train, y_test = y_all[:train_size], y_all[train_size:]
train_dates, test_dates = label_dates[:train_size], label_dates[train_size:]

# Reshape input to [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test  = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# 5. Build and Train LSTM Model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()
early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, callbacks=[early_stop])
# Now save the model to a file
model.save("lstm_v2_model.h5")

# 6. Make Predictions
y_pred = model.predict(X_test)

# Inverse transform
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate residuals
residuals = y_test_inv - y_pred_inv

# 7. Plot with Actual Dates

# Plot actual vs predicted prices
plt.figure(figsize=(12, 6))
plt.plot(test_dates, y_test_inv, color='blue', label='Actual Stock Price')
plt.plot(test_dates, y_pred_inv, color='red', label='Predicted Stock Price')
plt.title('Stock Price Prediction Using LSTM')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Plot residuals
plt.figure(figsize=(12, 6))
plt.plot(test_dates, residuals, color='green', label='Residuals')
plt.title('Residuals (Actual - Predicted)')
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.legend()
plt.show()
