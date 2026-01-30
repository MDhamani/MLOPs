import pickle
import os
import json
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import joblib
import argparse
import sys

sys.path.insert(0, os.path.abspath('..'))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()
    
    # Access the timestamp
    timestamp = args.timestamp
    try:
        model_version = f'model_{timestamp}_xgb_model'  # Use a timestamp as the version
        model = joblib.load(f'{model_version}.joblib')
    except:
        raise ValueError('Failed to load the latest model')
    
    # Load the Breast Cancer dataset
    try:
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split
        
        breast_cancer = load_breast_cancer()
        X = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
        y = pd.Series(breast_cancer.target)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    except:
        raise ValueError('Failed to load the data')
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    metrics = {
        "Accuracy": accuracy,
        "F1_Score": f1
    }
    
    # Save metrics to a JSON file
    if not os.path.exists('metrics/'): 
        os.makedirs("metrics/")
        
    with open(f'metrics/{timestamp}_metrics.json', 'w') as metrics_file:
        json.dump(metrics, metrics_file, indent=4)
    
    print(f"Model evaluation complete. Metrics saved to metrics/{timestamp}_metrics.json")
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
               
    
