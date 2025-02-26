# Stock Price Prediction with LSTM and Hyperparameter Tuning

Code implements a multi-step stock price forecasting model using Long Short-Term Memory (LSTM) neural networks with hyperparameter tuning via Keras Tuner.

## Overview

The model predicts the next 5 days of stock returns based on a 60-day lookback window using multiple features from historical stock data. A hyperparameter search optimizes the LSTM architecture for better predictions.

## Features

- Multi-step time series forecasting (5-day horizon)
- Multiple input features (Open, High, Low, Close, Volume, YTD Gains, Returns)
- Hyperparameter tuning with Keras Tuner
- Data preprocessing with MinMaxScaler
- Visualizations of predictions and historical data

## Dependencies

- numpy
- pandas
- matplotlib
- tensorflow
- scikit-learn
- keras-tuner

## Installation

```bash
pip install numpy pandas matplotlib tensorflow scikit-learn keras-tuner
```

## Usage

1. Prepare your historical stock data in CSV format (e.g., 'AAPL.csv')
2. Run the script:

```bash
python LSTM_tuned_predict.py
```

## Data Format

The script expects a CSV file with at least the following columns:
- Date
- Open
- High
- Low
- Close
- Volume
- YTD Gains

## Model Architecture

The LSTM model consists of:
- Two LSTM layers with tunable units (32-128)
- Dropout layers after each LSTM layer (rates 0.1-0.5)
- A Dense output layer predicting 5 future returns

## Hyperparameter Tuning

The script uses Keras Tuner's RandomSearch to optimize:
- Number of units in each LSTM layer
- Dropout rates

## Visualization

The script generates two plots:
1. Scaled future returns prediction
2. Historical close prices with forecasted future prices

## Customization

You can modify the following parameters:
- `time_step`: The lookback window (default: 60 days)
- `future_days`: The forecasting horizon (default: 5 days)
- `feature_cols`: The features used for prediction

## License

[MIT License](LICENSE)
