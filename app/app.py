import joblib
import pandas as pd
import logging
from flask import Flask, render_template, request, jsonify

from sklearn import set_config
from utils import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Initialize Flask app
app = Flask(__name__)


def load_models(preprocessing_path='../models/preprocessing_pipeline.joblib',
                model_path='../models/model.joblib'):
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


def prepare_input(request_data: dict) -> pd.DataFrame:
    """Convert request data to a DataFrame."""
    try:
        logger.info("Preparing input data...")
        input_data = {
            'id': 3,
            'CustomerId': int(request_data.get('CustomerId', 0)),
            'Surname': request_data.get('Surname', ''),
            'CreditScore': float(request_data['CreditScore']),
            'Geography': request_data['Geography'],
            'Gender': request_data['Gender'],
            'Age': float(request_data['Age']),
            'Tenure': float(request_data['Tenure']),
            'Balance': float(request_data['Balance']),
            'NumOfProducts': float(request_data['NumOfProducts']),
            'HasCrCard': float(request_data['HasCrCard']),
            'IsActiveMember': float(request_data['IsActiveMember']),
            'EstimatedSalary': float(request_data['EstimatedSalary']),
        }
        return pd.DataFrame([input_data])
    except Exception as e:
        logger.error(f"Error preparing input data: {str(e)}")
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
            '../models/data/playground-series-s4e1/test.csv')
        logger.info("Test data loaded successfully.")
        return test_data
    except Exception as e:
        logger.error(f"Error loading test data: {str(e)}")
        raise


@app.route('/', methods=['GET', 'POST'])
def home():
    set_config(transform_output='pandas')
    if request.method == 'POST':
        try:
            # Log received form data
            logger.info(f"Form input data: {request.form}")

            # Existing code for input_data creation
            input_data = {
                'id': 3,
                'CustomerId': int(request.form.get('CustomerId', 0)),
                'Surname': request.form.get('Surname', ''),
                'CreditScore': float(request.form['CreditScore']),
                'Geography': request.form['Geography'],
                'Gender': request.form['Gender'],
                'Age': float(request.form['Age']),
                'Tenure': float(request.form['Tenure']),
                'Balance': float(request.form['Balance']),
                'NumOfProducts': float(request.form['NumOfProducts']),
                'HasCrCard': float(request.form['HasCrCard']),
                'IsActiveMember': float(request.form['IsActiveMember']),
                'EstimatedSalary': float(request.form['EstimatedSalary']),
            }
            
            input_df = pd.DataFrame([input_data])

            # Call predict_churn
            churn_prob, prediction = predict_churn(
                input_df, preprocessor, model)
            logger.info(
                f"Churn probability: {churn_prob}, Prediction: {prediction}")

            return render_template('result.html', prediction=prediction,
                                   churn_prob=round(churn_prob * 100, 2),
                                   data=input_data)
        except Exception as e:
            logger.error(f"Error during prediction: {e}", exc_info=True)
            return render_template('error.html', error=str(e))
    return render_template('form.html', sample_data={})


@app.route('/use_sample', methods=['GET'])
def use_sample():
    """Endpoint for fetching a sample from the test dataset."""
    try:
        sample = test_data.sample(1).iloc[0].to_dict()
        return jsonify(sample)
    except Exception as e:
        logger.error(f"Error fetching sample data: {str(e)}")
        return jsonify({'error': str(e)})


def main():
    """Main execution function."""
    global preprocessor, model, test_data

    # Load models and test data
    preprocessor, model = load_models()
    test_data = load_test_data()

    # Run the Flask app
    app.run(debug=False)


if __name__ == "__main__":
    # preprocessor, model = load_models()
    # test_data = load_test_data()
    # test_sample = test_data.sample(1)
    # churn_prob, prediction = predict_churn(test_sample, preprocessor, model)
    # print(f"Test churn probability: {churn_prob}, Prediction: {prediction}")
    main()
