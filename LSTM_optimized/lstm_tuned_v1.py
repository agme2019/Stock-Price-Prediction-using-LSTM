import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt
import sys

# Redirect print statements to a log file
class Logger(object):
    def __init__(self, filename="model_output.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

sys.stdout = Logger("model_output.log")

# Function to create the model
def build_model(hp):
    model = Sequential()
    model.add(LSTM(units=hp.Int('units_1', min_value=32, max_value=512, step=32),
                   return_sequences=True, input_shape=(time_step, n_features)))
    model.add(Dropout(rate=hp.Float('dropout_1', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(LSTM(units=hp.Int('units_2', min_value=32, max_value=512, step=32)))
    model.add(Dropout(rate=hp.Float('dropout_2', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# 1. Load and Prepare the Data
data = pd.read_csv('AAPL.csv')
print(data.head())
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data.sort_index(inplace=True)  # Ensure chronological order

# Extract relevant columns
features = ['Open', 'High', 'Low', 'Close', 'Volume']
data = data[features]

# 2. Scale the Data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 3. Create Sequences
def create_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(len(dataset) - time_step):
        X.append(dataset[i : i + time_step])
        y.append(dataset[i + time_step, 3])  # Close price is the 4th column
    return np.array(X), np.array(y)

time_step = 60
X_all, y_all = create_dataset(scaled_data, time_step=time_step)

# We also need corresponding dates for the labels.
# The label for i-th sequence is at index (i + time_step).
all_dates = data.index
label_dates = all_dates[time_step:]  # Each label corresponds to this shifted index

# 4. Train-Test Split
train_size = int(len(X_all) * 0.8)
X_train, X_test = X_all[:train_size], X_all[train_size:]
y_train, y_test = y_all[:train_size], y_all[train_size:]
train_dates, test_dates = label_dates[:train_size], label_dates[train_size:]

# Reshape input to [samples, time steps, features]
n_features = X_all.shape[2]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], n_features)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], n_features)

# Hyperparameter tuning with Keras Tuner
tuner = kt.RandomSearch(build_model, objective='val_loss', max_trials=10, directory='my_dir', project_name='tune_stock_prediction')

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

tuner.search(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stop])
best_model = tuner.get_best_models(num_models=1)[0]

# Train the best model
best_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stop])

# Save the best model
best_model.save("lstm_v2_best_model.h5")

# Make Predictions
y_pred = best_model.predict(X_test)

# Inverse transform the scaled data
# We only want to inverse transform the 'Close' prices, not the entire dataset
close_scaler = MinMaxScaler(feature_range=(0, 1))
close_data = data[['Close']]
close_scaler.fit_transform(close_data)

# Inverse transform predictions and actual values
y_pred_inv = close_scaler.inverse_transform(y_pred)
y_test_inv = close_scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate residuals
residuals = y_test_inv - y_pred_inv

# Plot actual vs predicted prices
plt.figure(figsize=(12, 6))
plt.plot(test_dates, y_test_inv, color='blue', label='Actual Stock Price')
plt.plot(test_dates, y_pred_inv, color='red', label='Predicted Stock Price')
plt.title('Stock Price Prediction Using LSTM')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.savefig('stock_price_prediction.png')
plt.show()

# Plot residuals
plt.figure(figsize=(12, 6))
plt.plot(test_dates, residuals, color='green', label='Residuals')
plt.title('Residuals (Actual - Predicted)')
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.legend()
plt.savefig('residuals.png')
plt.show()

# Plot histogram of residuals
plt.figure(figsize=(12, 6))
plt.hist(residuals, bins=50, color='purple')
plt.title('Histogram of Residuals')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.savefig('histogram_of_residuals.png')
plt.show()

# Display summary statistics of residuals
print("Summary Statistics of Residuals:")
print(pd.DataFrame(residuals).describe())