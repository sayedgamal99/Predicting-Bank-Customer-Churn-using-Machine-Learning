# Core Libraries
import numpy as np
import pandas as pd
import logging
import warnings

# Scikit-learn
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn import set_config

# External Libraries for Encoding and Models
from category_encoders import TargetEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Serialization
import joblib



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

# Configure sklearn to output pandas DataFrames
set_config(transform_output='pandas')

# Model configurations
MODEL_PARAMS = {
    'xgboost': {
        'model': XGBClassifier(
            n_estimators=1500,
            learning_rate=0.009,
            n_jobs=-1,
            random_state=3,
            verbose=0
        ),
        'fit_params': {
            'early_stopping_rounds': 20,
            'verbose':0
        }
    },
    'lightgbm': {
        'model': LGBMClassifier(
            learning_rate=0.0088,
            max_depth=16,
            n_estimators=2000,
            num_leaves=16,
            random_state=3,
            verbose=0
        ),
        'fit_params': {
            
        }
    },
    'catboost': {
        'model': CatBoostClassifier(
            learning_rate=0.0009,
            iterations=9000,
            random_seed=3,
            verbose=0,
        ),
        'fit_params': {
            'verbose': 0,
            'early_stopping_rounds': 20,
        }
    }
}

# Define preprocessing pipelines for different column types
cat_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('encode', OneHotEncoder(handle_unknown='ignore', sparse=False, drop='first'))
])

num_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='median'))
])

target_enc_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('encode', TargetEncoder())
])

# Custom feature engineering functions
def ActiveCrCardFunc(d):
    """Create ActiveCrCard feature from HasCrCard and IsActiveMember"""
    d['ActiveCrCard'] = d['HasCrCard'] + (4 * d['IsActiveMember']) + 1
    return d[['ActiveCrCard']]

def TenurebyAgeFunc(d):
    """Create TenureAgeRatio feature and log transform Age"""
    d['Age'] = np.log(d['Age'])
    d['TenureAgeRatio'] = d['Tenure'] / d['Age']
    return d[['TenureAgeRatio']]

# Create feature-specific pipelines
ActiveCrCard_pipeline = make_pipeline(
    num_pipeline,
    FunctionTransformer(ActiveCrCardFunc, validate=False)
)

TenureAge_pipeline = make_pipeline(
    num_pipeline,
    FunctionTransformer(TenurebyAgeFunc, validate=False)
)

# Main preprocessing pipeline
preprocessing = ColumnTransformer([
    ('N', num_pipeline, ['CreditScore', 'Balance', 'EstimatedSalary', 'Tenure']),
    ('ActiveCrCard', ActiveCrCard_pipeline, ['IsActiveMember', 'HasCrCard']),
    ('Age', make_pipeline(num_pipeline, FunctionTransformer(func=np.log)), ['Age']),
    ('Cat', cat_pipeline, ['Geography', 'Gender', 'NumOfProducts']),
    ('Tar', target_enc_pipeline, ['Surname']),
], remainder='drop', verbose_feature_names_out=False)

# Final preprocessing pipeline with scaling
preprocessing = Pipeline([
    ('Preprocessing for columns', preprocessing),
    ('Scaling', StandardScaler())
])