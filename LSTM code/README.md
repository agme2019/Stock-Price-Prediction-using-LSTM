### Information on model (code : model_LSTM_v2.py)
This is the first LSTM code that was created. Approach, usage, results and drawbacks are discussed.
#### Introduction
This report details the process of building and training an LSTM (Long Short-Term Memory) model to predict stock prices, specifically using Apple's stock data (AAPL). The model utilizes historical closing prices to forecast future prices. The steps include data preparation, scaling, sequence creation, train-test split, model building, training, and evaluation.

#### Data Loading and Preprocessing

**Libraries Import:**
The code begins by importing essential libraries for data manipulation, visualization, and building the LSTM model:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
```

**Data Loading:**
The dataset is loaded from a CSV file named 'AAPL.csv' into a pandas DataFrame, and the first few rows are printed to verify the data structure:
```python
data = pd.read_csv('AAPL.csv')
print(data.head())
```

**Date Parsing and Indexing:**
The 'Date' column is converted to datetime format and set as the DataFrame index. Sorting ensures the data is in chronological order:
```python
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data.sort_index(inplace=True)
```

**Feature Extraction:**
Only the 'Close' column, representing the closing prices, is extracted for further analysis:
```python
close_data = data[['Close']]
```

#### Data Scaling

**Normalization:**
The closing prices are scaled to a range of 0 to 1 using MinMaxScaler. This normalization is crucial for LSTM models, as they are sensitive to the scale of input data:
```python
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_data)
```

#### Sequence Creation

**Sliding Window Technique:**
A function `create_dataset` is defined to generate sequences of a specified length (`time_step`). Each sequence consists of `time_step` consecutive closing prices (X), and the corresponding label (y) is the closing price immediately following this sequence:
```python
def create_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(len(dataset) - time_step):
        X.append(dataset[i : i + time_step, 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X_all, y_all = create_dataset(scaled_data, time_step=time_step)
```

**Obtaining Label Dates:**
To align the labels with their corresponding dates, the dates are shifted by `time_step`:
```python
all_dates = close_data.index
label_dates = all_dates[time_step:]
```

#### Train-Test Split

**Data Division:**
The dataset is split into training (80%) and testing (20%) sets. The corresponding dates for the labels are also split:
```python
train_size = int(len(X_all) * 0.8)
X_train, X_test = X_all[:train_size], X_all[train_size:]
y_train, y_test = y_all[:train_size], y_all[train_size:]
train_dates, test_dates = label_dates[:train_size], label_dates[train_size:]
```

**Reshaping Data for LSTM:**
The input data is reshaped to the format `[samples, time steps, features]` required by LSTM layers:
```python
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test  = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
```

#### Model Architecture

**Building the LSTM Model:**
A sequential LSTM model is built with the following architecture:
- First LSTM layer with 50 units and `return_sequences=True` to return the full sequence.
- Dropout layer with 20% dropout rate to prevent overfitting.
- Second LSTM layer with 50 units with `return_sequences=False` to return output for the last time step.
- Another Dropout layer with 20% dropout rate.
- Dense layer with 1 unit for the final output:
```python
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()
```

**Compilation:**
The model is compiled with Adam optimizer and mean squared error loss function.

**Early Stopping:**
Early stopping is implemented to halt training if the loss doesn't improve for 5 epochs:
```python
early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, callbacks=[early_stop])
model.save("lstm_v2_model.h5")
```

#### Model Training and Saving

**Training:**
The model is trained for up to 50 epochs with a batch size of 32. Early stopping prevents overfitting.

**Model Saving:**
The trained model is saved for future use.

#### Prediction and Visualization

**Prediction:**
The model predicts the closing prices for the test set. The predictions and actual values are scaled back to their original range:
```python
y_pred = model.predict(X_test)
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
```

**Visualization:**
A plot compares the actual and predicted prices, with dates on the x-axis, providing a visual assessment of the model's performance:
```python
plt.figure(figsize=(12, 6))
plt.plot(test_dates, y_test_inv, color='blue', label='Actual Stock Price')
plt.plot(test_dates, y_pred_inv, color='red', label='Predicted Stock Price')
plt.title('Stock Price Prediction Using LSTM')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
```
**Results:**
Actual vs Model
![LSTMv2_results](https://github.com/user-attachments/assets/c3491798-cc5c-4e6b-86c6-1c78d25a9c4b)
Residuals (deviation from actual)
![LSTMv2_residuals](https://github.com/user-attachments/assets/1a8bd8c9-fc71-4667-86d7-c2438f069b24)

### Summary

**Data Handling:**
The code efficiently handles time series data, ensuring chronological order and appropriate preprocessing.

**Model Architecture:**
A two-layer LSTM with dropout layers is used, which is a standard approach for time series forecasting.

**Evaluation:**
Visual evaluation is performed using a plot, which is a common method to assess model performance in time series analysis.

### Conclusion

The LSTM model attempts to predict the stock prices using historical data. The steps involved include data preprocessing, scaling, sequence generation, train-test split, model building, training, and evaluation. The model's performance is visualized through a plot comparing actual and predicted prices. From the performance evalution, it is observed that the model can deviate by +20/-10 from the actual stock price, which suggests that it is not a good model. This approach demonstrates the potential of LSTM networks for time series forecasting in financial markets.

### Future Improvements

**Performance Metrics:**
In addition to visual evaluation, calculating metrics like RMSE (Root Mean Squared Error) or MAE (Mean Absolute Error) would provide a quantitative assessment of the model's performance.

**Hyperparameter Tuning:**
Experimenting with different window sizes, the number of LSTM layers, and units could potentially improve the model's accuracy.

**Model Complexity:**
Exploring more complex architectures or different types of recurrent layers may yield better results.

This code serves as a foundation for stock price prediction using LSTM, with potential enhancements in areas like hyperparameter tuning, performance metrics, and model complexity.
