import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Define constants directly instead of importing
CITY = "karachi"
TARGET_COLUMN = "aqi_next_3days"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KarachiFeatureEngineer:
    def __init__(self):
        self.city = CITY
        self.feature_columns = []
    
    def create_features(self, raw_data):
        """Create features from raw AQI data for Karachi"""
        if isinstance(raw_data, dict):
            raw_data = pd.DataFrame([raw_data])
        
        if raw_data.empty:
            logger.error("No data provided for feature engineering")
            return None
            
        df = raw_data.copy()
        
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        
        # Cyclical time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        
        # AQI categories (for feature creation)
        df['aqi_category'] = pd.cut(df['aqi'], 
                                   bins=[0, 50, 100, 150, 200, 300, 500],
                                   labels=[1, 2, 3, 4, 5, 6])
        
        # Pollutant ratios
        df['pm_ratio'] = np.where(df['pm10'] > 0, df['pm25'] / df['pm10'], 0)
        df['no2_so2_ratio'] = np.where(df['so2'] > 0, df['no2'] / (df['so2'] + 1e-5), 0)
        
        # Weather interactions
        df['temp_humidity'] = df['temperature'] * df['humidity']
        df['wind_pressure'] = df['wind_speed'] * df['pressure']
        
        # Lag features (if we have historical data)
        if len(df) > 1:
            df['aqi_lag_1'] = df['aqi'].shift(1)
            df['pm25_lag_1'] = df['pm25'].shift(1)
        else:
            df['aqi_lag_1'] = df['aqi']
            df['pm25_lag_1'] = df['pm25']
        
        # Fill NaN values - FIXED version (remove deprecated method)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in df.columns:
                # Forward fill then backward fill
                df[col] = df[col].ffill().bfill()
        
        # Define feature columns (exclude metadata)
        exclude_cols = ['city', 'timestamp', 'dominant_pollutant', 'weather_description', 
                       'aqi_category', TARGET_COLUMN, 'insert_timestamp']
        
        self.feature_columns = [col for col in df.columns if col not in exclude_cols and col != 'aqi']
        
        logger.info(f"Created {len(self.feature_columns)} features for Karachi")
        return df
    
    def create_target(self, df, forecast_horizon=3):
        """Create target variable (AQI after 3 days)"""
        df = df.copy()
        
        # For demo: Create synthetic target based on current AQI with trend
        np.random.seed(42)
        
        # Simple prediction: AQI tends to persist with some variation
        df[TARGET_COLUMN] = df['aqi'] + np.random.normal(0, 15, len(df))
        df[TARGET_COLUMN] = df[TARGET_COLUMN].clip(0, 500)  # Keep in valid range
        
        return df
    
    def save_features_to_store(self, features_df, feature_group_name="karachi_aqi"):
        """Save engineered features to feature store"""
        try:
            # Import here to avoid circular imports
            from .feature_store import feature_store
            success = feature_store.insert_feature_data(features_df, feature_group_name)
            if success:
                logger.info("Features saved to feature store")
            return success
        except Exception as e:
            logger.error(f"Error saving features to store: {e}")
            return False

if __name__ == "__main__":
    # Test feature engineering
    from data_collection import KarachiAQICollector
    
    collector = KarachiAQICollector()
    current_data = collector.get_current_aqi()
    
    if current_data:
        engineer = KarachiFeatureEngineer()
        features_df = engineer.create_features(current_data)
        features_with_target = engineer.create_target(features_df)
        
        print("✅ Features created successfully!")
        print(f"Features: {engineer.feature_columns}")
        print(f"Target: {TARGET_COLUMN}")
        
        # Save to feature store
        engineer.save_features_to_store(features_with_target)