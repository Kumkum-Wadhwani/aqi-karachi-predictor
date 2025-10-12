import pandas as pd
import os
from datetime import datetime
import logging

# Remove the problematic import and define CITY directly
CITY = "karachi"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleFeatureStore:
    """Simple local feature store that mimics Hopsworks functionality"""
    
    def __init__(self, base_path="feature_store"):
        self.base_path = base_path
        self.feature_store_path = f"{base_path}/features"
        self.model_registry_path = f"{base_path}/models"
        
        # Create directories
        os.makedirs(self.feature_store_path, exist_ok=True)
        os.makedirs(self.model_registry_path, exist_ok=True)
        logger.info("Simple Feature Store initialized")
    
    def insert_feature_data(self, feature_data, feature_group_name="karachi_aqi"):
        """Insert feature data into feature store"""
        try:
            # Add metadata
            feature_data['insert_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            feature_data['city'] = CITY
            
            # Convert to DataFrame if it's a dictionary
            if isinstance(feature_data, dict):
                feature_data = pd.DataFrame([feature_data])
            
            # Save to CSV (append mode)
            filename = f"{self.feature_store_path}/{feature_group_name}.csv"
            
            if os.path.exists(filename):
                # Append to existing file
                existing_df = pd.read_csv(filename)
                combined_df = pd.concat([existing_df, feature_data], ignore_index=True)
                combined_df.to_csv(filename, index=False)
            else:
                # Create new file
                feature_data.to_csv(filename, index=False)
            
            logger.info(f"Feature data inserted into {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error inserting feature data: {e}")
            return False
    
    def get_training_data(self, feature_group_name="karachi_aqi", lookback_days=30):
        """Get training data from feature store"""
        try:
            filename = f"{self.feature_store_path}/{feature_group_name}.csv"
            
            if os.path.exists(filename):
                df = pd.read_csv(filename)
                
                # Convert timestamp
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                if 'insert_timestamp' in df.columns:
                    df['insert_timestamp'] = pd.to_datetime(df['insert_timestamp'])
                
                # Get recent data
                if 'timestamp' in df.columns:
                    cutoff_date = datetime.now() - pd.Timedelta(days=lookback_days)
                    recent_data = df[df['timestamp'] >= cutoff_date]
                else:
                    recent_data = df
                
                logger.info(f"Retrieved {len(recent_data)} records for training")
                return recent_data
            else:
                logger.warning("No feature data found")
                return None
                
        except Exception as e:
            logger.error(f"Error getting training data: {e}")
            return None
    
    def save_model(self, model, model_name="aqi_model", version=1):
        """Save model to model registry"""
        try:
            import joblib
            
            model_filename = f"{self.model_registry_path}/{model_name}_v{version}.pkl"
            joblib.dump(model, model_filename)
            
            logger.info(f"Model saved to {model_filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, model_name="aqi_model", version=1):
        """Load model from model registry"""
        try:
            import joblib
            
            model_filename = f"{self.model_registry_path}/{model_name}_v{version}.pkl"
            
            if os.path.exists(model_filename):
                model = joblib.load(model_filename)
                logger.info(f"Model loaded from {model_filename}")
                return model
            else:
                logger.warning(f"Model file not found: {model_filename}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None

# Global feature store instance
feature_store = SimpleFeatureStore()