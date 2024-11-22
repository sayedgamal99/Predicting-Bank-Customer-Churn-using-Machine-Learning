from utils import *


warnings.filterwarnings('ignore')
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_and_split_data(data_path='model/data/playground-series-s4e1/train.csv'):
    """Load and split data into train, validation, and test sets"""
    logger.info("Loading and splitting data")
    df = pd.read_csv(data_path)
    X, y = df.drop(columns=['Exited']), df['Exited']
    
    # Split into train and temp (for validation and test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=3
    )
    
    # Split temp into validation and test
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=322
    )
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def train_models(X_train, y_train, X_valid, y_valid):
    """Train all models"""
    trained_models = {}
    
    for name, config in MODEL_PARAMS.items():
        logger.info(f"Training {name} model")
        model = config['model']
        fit_params = config['fit_params']
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            **fit_params
        )
        trained_models[name] = model
    
    return trained_models


def main():
    # Load and split data
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_and_split_data()
    
    # Prepare data
    logger.info("Preparing data")
    train_prepared = preprocessing.fit_transform(X_train, y_train)
    valid_prepared = preprocessing.transform(X_valid)
    
    # Save preprocessing pipeline
    logger.info("Saving preprocessing pipeline")
    joblib.dump(preprocessing, 'model/preprocessing_pipeline.joblib')

    # Train models
    trained_models = train_models(train_prepared, y_train, valid_prepared, y_valid)
    
    # Create and save voting model
    logger.info("Creating and saving voting model")

    predictor = EnsemblePredictor(trained_models)
    voting_model = FunctionTransformer(predictor, validate=False)
    joblib.dump(voting_model, 'model/model.joblib')
    
    logger.info("Training complete!")

if __name__ == "__main__":
    main()