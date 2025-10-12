#!/usr/bin/env python3
"""
Quick demo script to test the AQI prediction system for Karachi
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_collection import KarachiAQICollector
from src.feature_engineering import KarachiFeatureEngineer
from src.feature_store import feature_store

def run_demo():
    print("🚀 Starting Karachi AQI Prediction System Demo")
    print("=" * 50)
    
    # Step 1: Data Collection
    print("\n1. 📊 Collecting AQI Data for Karachi...")
    collector = KarachiAQICollector()
    current_data = collector.get_current_aqi()
    
    if not current_data:
        print("❌ No data collected. Please check your API keys and internet connection.")
        return
    
    print("✅ Current AQI Data Collected:")
    print(f"   AQI: {current_data['aqi']}")
    print(f"   PM2.5: {current_data.get('pm25', 'N/A')}")
    print(f"   Temperature: {current_data.get('temperature', 'N/A')}")
    
    # Step 2: Feature Engineering
    print("\n2. 🔧 Engineering Features...")
    engineer = KarachiFeatureEngineer()
    features_df = engineer.create_features(current_data)
    features_with_target = engineer.create_target(features_df)
    
    print(f"✅ Created {len(engineer.feature_columns)} features")
    
    # Step 3: Store Features
    print("\n3. 💾 Storing Features...")
    success = engineer.save_features_to_store(features_with_target)
    
    if success:
        print("✅ Features stored successfully in feature store")
    
    # Step 4: Display AQI Status
    print("\n4. 📈 AQI Status:")
    aqi = current_data['aqi']
    if aqi <= 50:
        status = "Good ✅"
    elif aqi <= 100:
        status = "Moderate ✅"
    elif aqi <= 150:
        status = "Unhealthy for Sensitive Groups ⚠️"
    elif aqi <= 200:
        status = "Unhealthy 🚨"
    elif aqi <= 300:
        status = "Very Unhealthy 🚨"
    else:
        status = "Hazardous 💀"
    
    print(f"   Current AQI: {aqi} - {status}")
    
    print("\n🎉 Demo completed successfully!")
    print("\nNext steps:")
    print("1. Run 'python src/model_training.py' to train models")
    print("2. Run 'cd frontend && streamlit run app.py' to start web app")

if __name__ == "__main__":
    run_demo()