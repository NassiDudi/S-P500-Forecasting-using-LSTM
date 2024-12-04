# S&P500-Forecasting-using-LSTM

This repository contains a project for forecasting the S&P 500 ETF (SPY) using a Long Short-Term Memory (LSTM) neural network. The project leverages historical financial data to predict SPY prices, fine-tunes hyperparameters such as learning rate and batch size, and visualizes both training performance and prediction accuracy.

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Technologies Used](#technologies-used)  
3. [Dataset](#dataset)  
4. [Feature Engineering](#feature-engineering)  
5. [How to Run](#how-to-run)  
6. [Code Breakdown](#code-breakdown)  
7. [Visualization](#visualization)  
8. [Forecasting the Next 4 Days](#forecasting-the-next-4-days)  
9. [Results](#results)  

## Project Overview
This project uses a sequential LSTM-based neural network to predict future S&P 500 prices based on selected financial indicators. It includes early stopping for efficient training, hyperparameter tuning, and performance evaluation.

## Technologies Used
- Python 3.x
- TensorFlow/Keras for deep learning
- Matplotlib for data visualization
- NumPy for numerical computations
- Scikit-learn for data preprocessing

## Dataset
The dataset consists of historical data from Yahoo Finanace including:
- SPY (S&P 500 ETF)  
- AAPL (Apple Inc.)  
- XLF (Financial Select Sector SPDR Fund)  
- QQQ (Invesco QQQ Trust)  
- VXX (Volatility Index ETF)  
- DIA (Dow Jones ETF)
- SPY’s moving averages (MA10, MA50)  
- SPY’s Relative Strength Index (RSI)


## Feature Engineering
Feature engineering is crucial for enhancing the model's predictive capabilities. Here are the key features:

1. **SPY_MA10 and SPY_MA50**:  
   - 10-day and 50-day moving averages to smooth out short-term fluctuations and identify trends.
   
2. **SPY_RSI (Relative Strength Index)**:  
   - A momentum oscillator measuring the speed and change of price movements, signaling overbought or oversold conditions.
   
3. **Time-Series Lagging**:  
   - A window-based approach (size 10) to provide the model with historical context for predicting the next time step.

## How to Run
There are two ways you can run this project:

1. Using Google Colab (Recommended for easy setup)
Download the notebook:

Visit the GitHub repository and download the S&P500_Forecasting_using_LSTM.ipynb file.
Upload the notebook to Colab:

Open Google Colab.
Click on "File" > "Upload notebook" and select the downloaded .ipynb file.

2. Clone this repository:
   ```bash
   git clone https://github.com/NassiDudi/SPY-sp500--Forecasting-using-LSTM.git
   ```
   - Install the required packages
   - Use any IDE that supports Jupyter Notebooks

## Code Breakdown
1. **Data Preparation**:  
   `prepare_data()`processes historical prices into features and target variables for model training. It performs two critical tasks:
   - Normalization with `StandardScaler`: Data is normalized to have a mean of 0 and a standard deviation of 1. This standardization ensures all features are on the same scale, improving the convergence rate and         training stability of the LSTM model.

   - Sliding Window Creation for LSTM: The function creates sliding windows of data, where each window contains `window_size` consecutive time steps of input data (e.g., 10 time steps).
      - **X (features)**: Sequences of historical data (e.g., 10-day windows) provided as input to the LSTM.
      - **y (target)**: The target value (e.g., SPY price) corresponding to the next time step after each window.

2. **Model Building**:  
   `build_lstm()` creates an LSTM model with two LSTM layers, dropout for regularization, and dense output for regression.

3. **Training**:  
   Hyperparameter tuning involves adjusting learning rates and batch sizes with early stopping for optimal performance.
   The following hyperparameters are tuned during model training:
   - **Learning Rate**: [0.001, 0.01]
   - **Batch Size**: [32, 16]
   - **Early Stopping**: Monitors validation loss with `patience=3` to avoid overfitting.

4. **4-Day Forecast**:
   After training the model, the best-performing configuration(selcted by RMSE) is used to predict the next 4 days of SPY prices.
   
## Visualization:
1. **Correlation Heatmap**: Displays the correlation between selected financial indicators and tickers.
2. **Individual Ticker Graphs**: Time-series plots for each ticker used in the model (SPY, AAPL, XLF, QQQ, VXX, DIA).
3. **Best Model Configuration**: Displays the learning rate, batch size, and RMSE of the best model.
4. **Predicted vs Actual Prices Plot**: Compares the predicted SPY prices with the actual prices on the original scale.

## Results:
**Model Selection and Evaluation**
The best LSTM model was selected based on the lowest RMSE (Root Mean Squared Error) during the hyperparameter tuning process. Different combinations of learning rates and batch sizes were tested, with the model achieving the optimal balance between training and validation loss when the learning rate and batch size were finely adjusted.

**Forecasting Performance**
- The best model achieved a low RMSE, indicating that the predicted values closely match the actual SPY prices.
- Prediction vs. Actual: The predicted SPY prices for the four-day forecast align closely with the true prices, demonstrating the model’s effectiveness in capturing market trends. The visualization shows minimal    deviation between the actual and predicted values.
These results highlight the model's robustness in forecasting short-term SPY price movements, providing valuable insights into future market behavior.
