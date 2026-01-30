import pytest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from unittest.mock import patch, MagicMock

from xgboost import XGBClassifier
from src.train_model import download_data, preprocess_data, train_model

# ----------------- Test Download ----------------- #
# Test the download_data function to ensure it correctly downloads and returns data
def test_download_data():
    X, y = download_data()
    
    # Check if the data is downloaded correctly and matches expected formats
    assert isinstance(X, pd.DataFrame)  # X should be a DataFrame
    assert isinstance(y, pd.Series)     # y should be a Series
    assert not X.empty                  # X should not be empty
    assert not y.empty                  # y should not be empty
    assert X.shape[0] == y.shape[0]     # The number of rows in X and y should be the same

# ----------------- Test Preprocess ----------------- #
# Test the preprocess_data function to ensure it correctly preprocesses the data
def test_preprocess_data():
    X, y = download_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    # Assert that the preprocessing splits the data correctly
    assert X_train.shape[0] + X_test.shape[0] == X.shape[0]  # Rows in train and test should total original rows
    assert y_train.shape[0] + y_test.shape[0] == y.shape[0]  # Rows in train and test labels should total original labels
    assert X_train.shape[1] == X.shape[1]  # Number of features should remain the same

# ----------------- Test Train model ----------------- #
# Test the train_model function to ensure it correctly trains the model
def test_train_model():
    # Generate sample data for testing using Breast Cancer dataset features
    X = pd.DataFrame({
        'mean radius': [14.5, 18.2, 16.8, 12.3, 19.5],
        'mean texture': [18.5, 25.3, 20.1, 15.8, 28.5],
        'mean perimeter': [95.5, 115.8, 108.2, 82.3, 125.6],
        'mean area': [680.0, 950.0, 850.0, 520.0, 1100.0],
        'mean smoothness': [0.095, 0.105, 0.110, 0.080, 0.120],
        'mean compactness': [0.12, 0.18, 0.15, 0.08, 0.22],
        'mean concavity': [0.08, 0.15, 0.12, 0.03, 0.18],
        'mean concave points': [0.05, 0.10, 0.08, 0.02, 0.12],
        'mean symmetry': [0.18, 0.22, 0.20, 0.15, 0.25],
        'mean fractal dimension': [0.062, 0.068, 0.065, 0.055, 0.072],
        'radius error': [0.4, 0.7, 0.5, 0.3, 0.9],
        'texture error': [1.2, 2.1, 1.5, 0.8, 2.8],
        'perimeter error': [2.5, 4.2, 3.1, 1.8, 5.5],
        'area error': [25.0, 55.0, 35.0, 15.0, 75.0],
        'smoothness error': [0.008, 0.012, 0.010, 0.005, 0.015],
        'compactness error': [0.012, 0.025, 0.018, 0.008, 0.032],
        'concavity error': [0.015, 0.045, 0.030, 0.010, 0.055],
        'concave points error': [0.008, 0.018, 0.012, 0.005, 0.020],
        'symmetry error': [0.015, 0.035, 0.025, 0.010, 0.040],
        'fractal dimension error': [0.003, 0.008, 0.005, 0.002, 0.010],
        'worst radius': [16.5, 22.3, 20.1, 14.2, 25.8],
        'worst texture': [22.5, 35.2, 28.5, 18.3, 40.5],
        'worst perimeter': [110.5, 155.2, 135.8, 95.3, 175.5],
        'worst area': [820.0, 1350.0, 1100.0, 680.0, 1600.0],
        'worst smoothness': [0.13, 0.16, 0.15, 0.10, 0.18],
        'worst compactness': [0.25, 0.55, 0.40, 0.15, 0.65],
        'worst concavity': [0.20, 0.60, 0.40, 0.10, 0.75],
        'worst concave points': [0.10, 0.20, 0.15, 0.05, 0.25],
        'worst symmetry': [0.30, 0.50, 0.40, 0.20, 0.55],
        'worst fractal dimension': [0.10, 0.16, 0.13, 0.08, 0.18],
    })
    y = pd.Series([0, 0, 0, 1, 1])
    
    # Train the model using the sample data
    model = train_model(X, y)
    
    # Assertions to verify the model is trained correctly
    assert isinstance(model, XGBClassifier)  # Check if the returned model is of the correct type
    assert hasattr(model, 'predict')                  # Ensure the model has a predict method