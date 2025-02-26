In this code, both scikit-learn and Keras (TensorFlow's high-level API) are being used together for a stock price prediction model. Let me explain how each library is utilized:

## scikit-learn Usage

1. **Data Preprocessing with `MinMaxScaler`**:In this code, both scikit-learn and Keras (TensorFlow's high-level API) are being used together for a stock price prediction model. Let me explain how each library is utilized:

## scikit-learn Usage

1. **Data Preprocessing with `MinMaxScaler`**:
   ```python
   from sklearn.preprocessing import MinMaxScaler
   scaler = MinMaxScaler(feature_range=(0, 1))
   scaled_data = scaler.fit_transform(data)
   ```
   The `MinMaxScaler` is used to normalize the stock data features (Open, High, Low, Close, Volume) to a range between 0 and 1. This normalization helps the neural network converge faster and perform better.

2. **Another scaler is used specifically for the Close price**:
   ```python
   close_scaler = MinMaxScaler(feature_range=(0, 1))
   close_data = data[['Close']]
   close_scaler.fit_transform(close_data)
   ```
   This separate scaler is later used to inverse transform the predictions back to the original scale.

## Keras (TensorFlow) Usage

1. **Model Architecture with `Sequential` API**:
   ```python
   from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import LSTM, Dense, Dropout
   ```
   Keras is used to build an LSTM (Long Short-Term Memory) neural network for the time series forecasting task.

2. **Model Definition in the `build_model` function**:
   ```python
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
   ```
   This creates a sequential model with two LSTM layers, Dropout regularization, and a Dense output layer.

3. **Early Stopping with `EarlyStopping` callback**:
   ```python
   from tensorflow.keras.callbacks import EarlyStopping
   early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
   ```
   The EarlyStopping callback prevents overfitting by stopping training when validation loss stops improving.

4. **Hyperparameter Tuning with `keras_tuner`**:
   ```python
   import keras_tuner as kt
   tuner = kt.RandomSearch(build_model, objective='val_loss', max_trials=10, directory='my_dir', project_name='tune_stock_prediction')
   tuner.search(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stop])
   ```
   Keras Tuner is used to find the optimal hyperparameters for the LSTM model, testing different combinations of units and dropout rates.

5. **Model Training and Prediction**:
   ```python
   best_model = tuner.get_best_models(num_models=1)[0]
   best_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stop])
   y_pred = best_model.predict(X_test)
   ```
   The best model from the hyperparameter search is trained and used to make predictions on the test data.

6. **Model Persistence**:
   ```python
   best_model.save("lstm_v2_best_model.h5")
   ```
   The trained model is saved to disk for future use.

The integration of these libraries shows a typical machine learning workflow: scikit-learn handles the data preprocessing and transformation aspects, while Keras provides the deep learning architecture for the time series prediction task. The code demonstrates how these libraries can be effectively combined to build a complete machine learning pipeline.
   ```python
   from sklearn.preprocessing import MinMaxScaler
   scaler = MinMaxScaler(feature_range=(0, 1))
   scaled_data = scaler.fit_transform(data)
   ```
   The `MinMaxScaler` is used to normalize the stock data features (Open, High, Low, Close, Volume) to a range between 0 and 1. This normalization helps the neural network converge faster and perform better.

2. **Another scaler is used specifically for the Close price**:
   ```python
   close_scaler = MinMaxScaler(feature_range=(0, 1))
   close_data = data[['Close']]
   close_scaler.fit_transform(close_data)
   ```
   This separate scaler is later used to inverse transform the predictions back to the original scale.

## Keras (TensorFlow) Usage

1. **Model Architecture with `Sequential` API**:
   ```python
   from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import LSTM, Dense, Dropout
   ```
   Keras is used to build an LSTM (Long Short-Term Memory) neural network for the time series forecasting task.

2. **Model Definition in the `build_model` function**:
   ```python
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
   ```
   This creates a sequential model with two LSTM layers, Dropout regularization, and a Dense output layer.

3. **Early Stopping with `EarlyStopping` callback**:
   ```python
   from tensorflow.keras.callbacks import EarlyStopping
   early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
   ```
   The EarlyStopping callback prevents overfitting by stopping training when validation loss stops improving.

4. **Hyperparameter Tuning with `keras_tuner`**:
   ```python
   import keras_tuner as kt
   tuner = kt.RandomSearch(build_model, objective='val_loss', max_trials=10, directory='my_dir', project_name='tune_stock_prediction')
   tuner.search(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stop])
   ```
   Keras Tuner is used to find the optimal hyperparameters for the LSTM model, testing different combinations of units and dropout rates.

5. **Model Training and Prediction**:
   ```python
   best_model = tuner.get_best_models(num_models=1)[0]
   best_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stop])
   y_pred = best_model.predict(X_test)
   ```
   The best model from the hyperparameter search is trained and used to make predictions on the test data.

6. **Model Persistence**:
   ```python
   best_model.save("lstm_v2_best_model.h5")
   ```
   The trained model is saved to disk for future use.

The integration of these libraries shows a typical machine learning workflow: scikit-learn handles the data preprocessing and transformation aspects, while Keras provides the deep learning architecture for the time series prediction task. The code demonstrates how these libraries can be effectively combined to build a complete machine learning pipeline.
