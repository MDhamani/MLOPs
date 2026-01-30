import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import joblib

# Function to download the Breast Cancer dataset from the sklearn library
def download_data():
    from sklearn.datasets import load_breast_cancer
    breast_cancer = load_breast_cancer()  # Load the Breast Cancer dataset
    features = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)  # Convert the breast cancer data to a DataFrame
    target = pd.Series(breast_cancer.target)  # Convert the breast cancer target to a Series
    return features, target

# Function to preprocess the downloaded data by splitting it into training and testing sets
def preprocess_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Function to train a XGBoost model using the preprocessed training data
def train_model(X_train, y_train):
    model = XGBClassifier(n_estimators=100, random_state=42, verbosity=0, base_score=0.5, use_label_encoder=False, eval_metric='logloss')  # Initialize XGBoost with specific parameters
    model.fit(X_train, y_train)  # Fit the model to the training data
    return model

# Function to save the trained model locally
def save_model(model, model_filename):
    joblib.dump(model, model_filename)  # Save the model locally using joblib

# Main function that orchestrates downloading data, training the model, and saving it
def main():
    # Download and preprocess data
    X, y = download_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    # Train and evaluate the model
    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model accuracy: {accuracy}')
    
    # Save the model locally
    save_model(model, "model.joblib")
    print("Model saved to model.joblib")

# Conditional to ensure the main function runs only if the script is executed directly
if __name__ == "__main__":
    main()