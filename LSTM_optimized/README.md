# LSTM Model Tuning

This repository provides a deep learning model using LSTM (Long Short-Term Memory) networks to predict stock prices based on historical data. It includes data preprocessing, hyperparameter tuning, model training, and evaluation.

## Key Features

- **LSTM Architecture**: Utilizes a stacked LSTM architecture with dropout layers for regularization.
- **Hyperparameter Tuning**: Employs Keras Tuner for efficient hyperparameter optimization.
- **Early Stopping**: Implements early stopping to prevent overfitting during training.
- **Data Scaling**: Uses Min-Max Scaler for normalizing the input data.
- **Visualization**: Generates plots for actual vs predicted prices, residuals, and their distribution.
- **Model Saving**: Saves the best-performing model for future use.

The script will:
- Load and preprocess the data
- Tune hyperparameters using Keras Tuner
- Train the model with early stopping
- Save the best model
- Generate prediction plots and residual analysis

## Code Structure

The code is organized into the following main sections:

1. **Data Loading and Preprocessing**:
   - Loads the stock data
   - Scales the data using Min-Max Scaler
   - Creates training and testing datasets

2. **Model Building**:
   - Defines an LSTM model architecture with hyperparameter tuning

3. **Hyperparameter Tuning**:
   - Uses Keras Tuner to find optimal model parameters

4. **Model Training**:
   - Trains the model with early stopping
   - Saves the best-performing model

5. **Model Evaluation**:
   - Makes predictions on the test set
   - Calculates residuals
   - Visualizes actual vs predicted prices
   - Plots residual distribution


## Results

The script generates the following visualizations:
1. `stock_price_prediction.png`: A plot showing the actual vs predicted stock prices.
2. `residuals.png`: A plot of the residuals (actual - predicted).
3. `histogram_of_residuals.png`: A histogram of the residuals.

Additionally, it outputs summary statistics of the residuals.

## Limitations

- The model is trained on historical data and assumes that future market behavior will follow similar patterns.
- It does not account for external factors like economic indicators, news events, or company announcements.
- The performance may vary depending on the quality and length of the input data.

