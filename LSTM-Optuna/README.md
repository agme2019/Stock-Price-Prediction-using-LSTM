# LSTM Stock Price Prediction with Optuna Hyperparameter Optimization

This project implements a stock price prediction model using Long Short-Term Memory (LSTM) neural networks with hyperparameter optimization via Optuna.

## Overview

The model predicts the next day's closing price based on a 60-day lookback window using multiple features from historical stock data. Optuna is used to find the optimal hyperparameters for the LSTM architecture.

## Features

- Time series forecasting of stock prices
- Multiple input features (Open, High, Low, Close, Volume)
- Hyperparameter optimization with Optuna
- Data preprocessing with MinMaxScaler
- Early stopping to prevent overfitting
- Comprehensive visualization of predictions and residuals
- Logging functionality to record model output

## Dependencies

- numpy
- pandas
- matplotlib
- tensorflow
- scikit-learn
- optuna

## Installation

```bash
pip install numpy pandas matplotlib tensorflow scikit-learn optuna
```

## Usage

1. Prepare your historical stock data in CSV format (e.g., 'AAPL.csv')
2. Run the script:

```bash
python stock_prediction_optuna.py
```

## Data Format

The script expects a CSV file with at least the following columns:
- Date
- Open
- High
- Low
- Close
- Volume

## Model Architecture

The LSTM model consists of:
- Three LSTM layers with tunable units (50-200)
- Dropout layers after each LSTM layer (rates 0.2-0.5)
- A Dense output layer for price prediction

## Hyperparameter Tuning

The script uses Optuna to optimize:
- Number of units in each LSTM layer
- Dropout rates
- Batch size
- Number of training epochs

## Visualization

The script generates three plots:
1. Actual vs. predicted stock prices ![LSTM OPTUNA RESULTS](https://github.com/user-attachments/assets/4275905c-bd5f-4276-ae8f-e0b475cc36ce)

2. Residuals over time ![LSTM OPTUNA RESIDUALS](https://github.com/user-attachments/assets/658c5cc6-8276-48ee-b771-b24f3fedad84)

3. Histogram of residuals ![LSTM OPTUNA HISTOGRAM](https://github.com/user-attachments/assets/fced76f7-7f59-4683-b4f1-44cdb080fe82)


All plots are automatically saved as PNG files.

## Logging

Model outputs and statistics are logged to a file named "model_output.log" for later reference.

## Reproducibility

Random seeds are set for both NumPy and TensorFlow to ensure reproducible results.

## Customization

You can modify the following parameters:
- `time_step`: The lookback window (default: 60 days)
- `n_trials`: The number of Optuna trials for hyperparameter optimization

## License

[MIT License](LICENSE)
