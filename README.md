# Stock-Price-Prediction-using-LSTM

---

# Multi-Step LSTM for Stock Forecasting

This project demonstrates a multi-step Long Short-Term Memory (LSTM) model to forecast stock returns and reconstruct future stock prices. The model leverages multiple features from historical stock data and uses hyperparameter tuning with Keras Tuner to optimize performance. The goal is to predict the next few days of returns (and thereby future Close prices) using a 60-day lookback window.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Data Preprocessing](#data-preprocessing)
  - [Multi-Step Forecasting](#multi-step-forecasting)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
- [Installation & Requirements](#installation--requirements)
- [Usage](#usage)
- [Results & Visualization](#results--visualization)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Overview

In this project, we build an LSTM-based forecasting model to predict future stock returns. By forecasting returns directly, the model aims to overcome the challenges of non-stationarity in raw stock prices. We use a multi-step sequence-to-sequence approach, predicting several future days in one forward pass. Additionally, the project demonstrates the use of Keras Tuner to automatically search for the best model hyperparameters, such as the number of LSTM units, dropout rates, batch size, and epochs.

## Features

- **Multivariate Modeling:**  
  Uses several stock features (Open, High, Low, Close, Volume, YTD Gains, and Returns) for a richer understanding of market dynamics.
  
- **Target Transformation:**  
  Predicts daily returns rather than raw prices, making the target more stationary and suitable for modeling.
  
- **Direct Multi-Step Forecasting:**  
  Implements a sequence-to-sequence LSTM architecture that outputs a vector of future returns (e.g., next 5 days) in a single prediction.
  
- **Hyperparameter Tuning:**  
  Integrates Keras Tuner (Random Search) to automatically optimize hyperparameters for improved model performance.
  
- **Visualization:**  
  Plots model predictions alongside historical data and compares predicted versus true future returns.

## Dataset

The project uses historical stock data, for example, from AAPL. The dataset should include columns such as:
- `Date`
- `Open`
- `High`
- `Low`
- `Close`
- `Volume`
- `Symbol`
- `YTD Gains`

Additionally, the project calculates daily returns using the percentage change in the Close price. You can use your own dataset or download a dataset from Kaggle. I used this link : https://www.kaggle.com/datasets/adhoppin/financial-data?resource=download

## Methodology
The final prediction model is here : [a link]([https://github.com/user/repo/blob/branch/other_file.md](https://github.com/agme2019/Stock-Price-Prediction-using-LSTM/tree/main/LSTM-predict))
Initial model :
Optimized model (tuned) :
Optuna model (did not work very well)




### Data Preprocessing

- **Data Loading:**  
  The CSV file is loaded, and the Date column is set as the index. Data is sorted in chronological order.
  
- **Feature Engineering:**  
  A new column `Returns` is created using the percentage change in Close prices.
  
- **Scaling:**  
  All features are scaled using `MinMaxScaler` to a range between 0 and 1, which is crucial for training deep learning models.

### Multi-Step Forecasting

- **Dataset Creation:**  
  A sliding window approach is used where a 60-day lookback window is used to predict the next 5 days of returns.  
  The helper function `create_multi_step_dataset` constructs the input (`X`) and target (`y`) datasets.
  
- **Reconstruction of Prices:**  
  Predicted returns are used to reconstruct future Close prices starting from the last known value, enabling a visual overlay on historical price data.

### Hyperparameter Tuning

- **Model Building Function:**  
  The LSTM model architecture is defined in a function that accepts hyperparameters (number of LSTM units, dropout rates) as arguments.
  
- **Keras Tuner Integration:**  
  Keras Tuner is used with a Random Search strategy to explore different hyperparameter configurations. The tuner evaluates models using a validation split and selects the best hyperparameters based on the lowest validation loss.

## Results
Future Returns Prediction (Scaled)
![Figure_1](https://github.com/user-attachments/assets/57b54cc6-0a8d-4596-9b8d-3e1a91d71ef8)
Forecasted AAPL stock price for next 5 days
![Figure_2](https://github.com/user-attachments/assets/d5a34b1a-51cb-4914-a771-c2cbcf9e3105)


## Installation & Requirements

### Prerequisites

- Python 3.7 or higher
- pip

### Required Packages

Install the required packages using pip:

```bash
pip install numpy pandas matplotlib tensorflow scikit-learn keras-tuner
```
## Usage

1. **Clone the repository**:
    ```bash
    git clone https://github.com/agme2019/Stock-Price-Prediction-using-LSTM.git
    cd Stock-Price-Prediction-using-LSTM
    ```

2. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the script**:
    ```python
    python <code>.py
    ```

## More information :

1. **Prepare the Dataset:**  
   Place your `AAPL.csv` file (or another stock CSV) in the project directory. Ensure the CSV includes the required columns.

2. **Run the Code:**  
   The repository contains python scripts (e.g., `<code>.py`) that:
   - Loads and preprocesses the data
   - Creates the multi-step forecasting dataset
   - Performs hyperparameter tuning using Keras Tuner
   - Trains the final model
   - Plots the forecasted versus true returns and reconstructed future Close prices

   To run the script, simply execute:
   
   ```bash
   python <code>.py
   ```

3. **View Results:**  
   The script outputs the best hyperparameters, training history, and displays plots that show the predicted future returns and the reconstructed future Close prices alongside historical data.

## Results & Visualization

The project provides visualizations to help understand the model's performance:
- **Forecasted Returns:**  
  A plot comparing the true and predicted scaled returns for a test sample.
  
- **Reconstructed Future Prices:**  
  Using the predicted returns, the future Close prices are reconstructed and plotted alongside historical Close prices for a visual comparison.

## Future Work

- **Feature Expansion:**  
  Integrate additional features such as technical indicators, macroeconomic variables, or sentiment analysis.
  
- **Multi-Step Model Enhancement:**  
  Experiment with deeper or alternative architectures (e.g., GRU, Seq2Seq with Attention) to further improve forecasting accuracy.
  
- **Advanced Validation Techniques:**  
  Implement walk-forward validation to simulate real-world forecasting scenarios.
  
- **Deployment:**  
  Develop a web-based dashboard (using Streamlit or Flask) to interactively display predictions and performance metrics.
    
## Contributing

Contributions, suggestions, and improvements are welcome! Feel free to fork the repository and submit a pull request. For major changes, please open an issue first to discuss your ideas.

## License

This project is open-source and available under the [MIT License](LICENSE).

---

