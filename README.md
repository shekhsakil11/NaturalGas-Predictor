# 🌍 Natural Gas Price Predictor

*A machine learning solution for forecasting natural gas prices with seasonal trend analysis*

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Polynomial%20Regression-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## Project Preview

![Natural Gas Price Prediction]([Assets/Screenshot 2025-11-07 162256.png]

## Project Overview

**Predicting natural gas prices for any future date using historical data and seasonal pattern analysis.** This tool helps energy traders and analysts make informed decisions by forecasting price trends up to one year in advance.

## Business Problem

Commodity storage contracts require accurate price forecasting to optimize buying (injection) and selling (withdrawal) timing. Traditional methods rely on limited monthly snapshots, but this solution provides **daily granularity** and captures **seasonal patterns** for better trading strategies.

## Key Features

- **Price Prediction**: Get natural gas prices for any past or future date
- **Seasonal Analysis**: Identify winter vs summer price patterns
- **1-Year Forecast**: Extended predictions for long-term planning
- **Interactive Tool**: User-friendly command-line interface
- **Data Visualization**: Comprehensive charts and trend analysis

## Key Insights

![Seasonal Pattern Analysis](https://github.com/shekhsakil11/NaturalGas-Predictor/blob/f696e5b151ae511a3be6660f2cffb092d585238c/Assets/Screenshot%202025-11-07%20162307.png)g)

- **Winter Premium**: Prices typically peak in December-January (heating season)
- **Summer Discount**: Lowest prices occur in May-June (low demand period)
- **Price Volatility**: Average seasonal spread of $1.50-$2.00 between high/low months
- **Upward Trend**: Overall increasing price trend from 2020-2024

## Tech Stack

- **Python 3.8+**
- **Data Analysis**: pandas, numpy
- **Machine Learning**: scikit-learn
- **Visualization**: matplotlib
- **Date Handling**: datetime

## Model Performance

![Model Performance Comparison](assets/model-performance.png)

Our polynomial regression model achieves **89% accuracy (R² score)** in predicting natural gas prices, significantly outperforming baseline models.

## Project Workflow

![Project Workflow](assets/workflow.png)

## Installation & Quick Start

### Method 1: Local Setup
```bash
# 1. Clone repository
git clone https://github.com/shekhsakil11/NaturalGas-Predictor.git
cd NaturalGas-Predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the predictor
python gas_price_predictor.py
