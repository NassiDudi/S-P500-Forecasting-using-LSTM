# S&P500-Forecasting-using-LSTM

This repository contains a project for forecasting the S&P 500 ETF (SPY) using a Long Short-Term Memory (LSTM) neural network. The project leverages historical financial data to predict SPY prices, fine-tunes hyperparameters such as learning rate and batch size, and visualizes both training performance and prediction accuracy.

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Technologies Used](#technologies-used)  
3. [Dataset](#dataset)  
4. [Feature Engineering](#feature-engineering)  
5. [How to Run](#how-to-run)  
6. [Code Breakdown](#code-breakdown)  
7. [Results](#results)  
8. [Forecasting the Next 4 Days](#forecasting-the-next-4-days)  
9. [Future Enhancements](#future-enhancements)  

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

## Feature Engineering
Feature engineering is crucial for enhancing the model's predictive capabilities. Here are the key features:

1. **SPY_MA10 and SPY_MA50**:  
   - 10-day and 50-day moving averages to smooth out short-term fluctuations and identify trends.
   
2. **SPY_RSI (Relative Strength Index)**:  
   - A momentum oscillator measuring the speed and change of price movements, signaling overbought or oversold conditions.
   
3. **Time-Series Lagging**:  
   - A window-based approach (size 10) to provide the model with historical context for predicting the next time step.

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/username/repository-name.git
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter notebook or Python script:
   ```bash
   jupyter notebook S&P500_Forecasting_using_LSTM.ipynb
