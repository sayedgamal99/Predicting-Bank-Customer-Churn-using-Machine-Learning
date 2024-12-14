import numpy as np
import pandas as pd
import logging
import warnings
import joblib

from sklearn import set_config


set_config(transform_output='pandas')
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnsemblePredictor:
    """Class to handle ensemble predictions"""

    def __init__(self, models):
        self.models = models

    def predict(self, X):
        predictions = []
        for model in self.models.values():
            pred = model.predict_proba(X)[:, 1]
            predictions.append(pred)
        return np.mean(predictions, axis=0)

    def __call__(self, X):
        return self.predict(X)


def ActiveCrCardFunc(d):
    d['ActiveCrCard'] = d['HasCrCard'] + (4 * d['IsActiveMember']) + 1
    return d[['ActiveCrCard']]


def TenurebyAgeFunc(d):
    d['Age'] = np.log(d['Age'])
    d['TenureAgeRatio'] = d['Tenure'] / d['Age']
    return d[['TenureAgeRatio']]




def load_models(preprocessing_path='./models/preprocessing_pipeline.joblib',
                model_path='./models/model.joblib'):
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

def predict_churn(input_df: pd.DataFrame, preprocessor, model) -> tuple:
    """Predict churn probability."""
    try:
        logger.info("Processing input data with the preprocessor...")
        processed_data = preprocessor.transform(input_df)

        # Predict churn probability
        logger.info("Making predictions...")
        predictions = model.transform(processed_data)
        churn_prob = predictions[0]  # Assuming the model outputs probabilities
        prediction = 'Churn' if churn_prob > 0.5 else 'Not Churn'
        return churn_prob, prediction
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise


def load_test_data():
    """Load test data for the sample feature."""
    try:
        logger.info("Loading test data...")
        test_data = pd.read_csv(
            './models/data/playground-series-s4e1/test.csv')
        logger.info("Test data loaded successfully.")
        return test_data
    except Exception as e:
        logger.error(f"Error loading test data: {str(e)}")
        raise

