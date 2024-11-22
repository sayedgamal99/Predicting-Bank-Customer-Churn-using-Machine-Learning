import pandas as pd
import numpy as np
import joblib
import logging

from utils import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_models(preprocessing_path='model/preprocessing_pipeline.joblib',
                model_path='model/model.joblib'):
    """Load the preprocessing pipeline and the model"""
    logger.info("Loading models...")
    try:
        preprocessing = joblib.load(preprocessing_path)
        model = joblib.load(model_path)
        logger.info("Models loaded successfully")
        return preprocessing, model
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

def predict_churn(data: pd.DataFrame, preprocessing, model) -> np.ndarray:
    """
    Make predictions on new data
    
    Args:
        data: DataFrame with the same columns as training data (except target)
        preprocessing: Loaded preprocessing pipeline
        model: Loaded prediction model
    
    Returns:
        Array of churn probabilities
    """
    try:
        # Apply preprocessing
        logger.info("Preprocessing data...")
        processed_data = preprocessing.transform(data)
        
        # Make predictions
        logger.info("Making predictions...")
        predictions = model.transform(processed_data)
        
        return predictions
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise

def main():
    # Load the models
    preprocessing, model = load_models()
    
    # Load your test data
    logger.info("Loading test data...")
    test_data = pd.read_csv('model/data/playground-series-s4e1/test.csv')
    
    # Make predictions
    predictions = predict_churn(test_data, preprocessing, model)
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        'id': range(len(predictions)),
        'prediction': predictions
    })
    
    # Save predictions
    logger.info("Saving predictions...")
    submission.to_csv('predictions.csv', index=False)
    logger.info("Predictions saved to predictions.csv")
    
    # Optional: Print some basic statistics
    logger.info(f"Average churn probability: {predictions.mean():.3f}")
    logger.info(f"Number of high-risk customers (>50% probability): {(predictions > 0.5).sum()}")


    logger.info(f"Test Completed Successfully.")

if __name__ == "__main__":
    main()