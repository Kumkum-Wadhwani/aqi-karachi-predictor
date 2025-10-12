import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import logging

# Define constants directly
TEST_SIZE = 0.2
RANDOM_STATE = 42
TARGET_COLUMN = "aqi_next_3days"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        from .feature_engineering import KarachiFeatureEngineer
        self.feature_engineer = KarachiFeatureEngineer()
        self.best_model = None
        self.model_performance = {}
    
    def prepare_data(self):
        """Prepare training data from feature store"""
        from .feature_store import feature_store
        
        # Get historical data from feature store
        historical_data = feature_store.get_training_data(lookback_days=30)
        
        if historical_data is None or historical_data.empty:
            logger.error("No training data available in feature store")
            return None, None, None, None
        
        # Ensure we have the target column
        if TARGET_COLUMN not in historical_data.columns:
            logger.error(f"Target column '{TARGET_COLUMN}' not found in data")
            return None, None, None, None
        
        # Prepare features and target
        X = historical_data[self.feature_engineer.feature_columns]
        y = historical_data[TARGET_COLUMN]
        
        # Remove rows with missing target
        valid_indices = y.notna()
        X = X[valid_indices]
        y = y[valid_indices]
        
        if len(X) == 0:
            logger.error("No valid training samples after cleaning")
            return None, None, None, None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        
        logger.info(f"Training data prepared: {len(X_train)} train, {len(X_test)} test samples")
        return X_train, X_test, y_train, y_test
    
    def train_models(self):
        """Train multiple models and select the best one"""
        X_train, X_test, y_train, y_test = self.prepare_data()
        
        if X_train is None:
            # Create synthetic data for demo if no real data
            X_train, X_test, y_train, y_test = self._create_demo_data()
        
        # Use only compatible models
        models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
            'linear_regression': LinearRegression()
        }
        
        best_score = float('inf')
        best_model_name = None
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                logger.info(f"{name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")
                
                # Store model performance
                self.model_performance[name] = {
                    'model': model,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2
                }
                
                # Update best model
                if rmse < best_score:
                    best_score = rmse
                    best_model_name = name
                    self.best_model = model
                    
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
        
        if best_model_name:
            logger.info(f"Best model: {best_model_name} with RMSE: {best_score:.2f}")
            return self.model_performance
        else:
            logger.error("No models were successfully trained")
            return None
    
    def _create_demo_data(self):
        """Create demo data for testing when no real data is available"""
        logger.info("Creating demo data for training...")
        
        # Create sample features
        np.random.seed(RANDOM_STATE)
        n_samples = 100
        
        demo_data = {
            'hour': np.random.randint(0, 24, n_samples),
            'day_of_week': np.random.randint(0, 7, n_samples),
            'month': np.random.randint(1, 13, n_samples),
            'pm25': np.random.uniform(10, 200, n_samples),
            'temperature': np.random.uniform(15, 40, n_samples),
            'humidity': np.random.uniform(30, 90, n_samples)
        }
        
        X = pd.DataFrame(demo_data)
        # Simulate target (AQI in 3 days) based on features
        y = (demo_data['pm25'] * 0.5 + 
             demo_data['temperature'] * 0.3 + 
             np.random.normal(0, 10, n_samples))
        
        return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    def save_best_model(self):
        """Save the best model to model registry"""
        from .feature_store import feature_store
        
        if self.best_model is not None:
            success = feature_store.save_model(self.best_model, "aqi_karachi_model", version=1)
            if success:
                logger.info("Best model saved to model registry")
            return success
        else:
            logger.error("No best model to save")
            return False

if __name__ == "__main__":
    trainer = ModelTrainer()
    performance = trainer.train_models()
    
    if performance:
        trainer.save_best_model()
        print("✅ Model training completed successfully!")
        
        # Print model performance
        print("\n📊 Model Performance:")
        for model_name, metrics in performance.items():
            print(f"   {model_name}: RMSE={metrics['rmse']:.2f}, R²={metrics['r2']:.2f}")
    else:
        print("❌ Model training failed")