# 🌫️ Karachi Air Quality Index (AQI) Predictor

A complete machine learning system for real-time AQI monitoring and 3-day forecasting in Karachi, Pakistan.

![AQI Dashboard](https://img.shields.io/badge/AQI-Monitoring-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![Streamlit](https://img.shields.io/badge/Web%20App-Streamlit-red)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-orange)

## 🚀 Live Demo

**🌐 Live Web App:** [View Live Dashboard](https://aqi-karachi-predictor.streamlit.app/)

## 📊 Project Overview

This project implements an end-to-end AQI prediction system that:
- 📡 Collects real-time air quality data from Karachi
- 🤖 Trains machine learning models for 3-day forecasts
- 📈 Provides interactive visualizations and health recommendations
- 🔄 Automates data collection via CI/CD pipeline
- 🌍 Deploys as a live web application

## 🛠️ Technical Features

### Data Pipeline
- **Real-time Data Collection**: AQICN.org API integration
- **Automated Processing**: Hourly data collection via GitHub Actions
- **Feature Store**: Local storage for historical data
- **Data Validation**: Quality checks and error handling

### Machine Learning
- **Multiple Models**: Random Forest, Linear Regression, Gradient Boosting
- **Smart Forecasting**: 3-day AQI predictions
- **Performance Monitoring**: RMSE, MAE, R² metrics
- **Model Retraining**: Automated daily training

### Web Dashboard
- **Real-time Monitoring**: Live AQI data from Karachi
- **Interactive Charts**: Plotly visualizations
- **Health Recommendations**: Personalized safety guidelines
- **Mobile Responsive**: Works on all devices

## 🏗️ System Architecture

Raw Data → Feature Engineering → ML Training → Web Dashboard
↓ ↓ ↓ ↓
AQICN API Feature Store Model Registry Streamlit
↓ ↓ ↓ ↓
GitHub Actions → Automated Pipeline → Live Predictions


## 🚀 Quick Start

### Local Development
```bash
# Clone repository
git clone https://github.com/Kumkum-Wadhwani/aqi-karachi-predictor.git

# Install dependencies
pip install -r requirements.txt

# Run data collection
python run_demo.py

# Start web app
cd frontend
streamlit run app.py
