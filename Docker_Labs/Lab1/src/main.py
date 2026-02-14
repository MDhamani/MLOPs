# Import necessary libraries
import logging
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logger.info("Starting model training pipeline...")
    
    # Load the Breast Cancer dataset
    logger.info("Loading Breast Cancer dataset...")
    breast_cancer = load_breast_cancer()
    X, y = breast_cancer.data, breast_cancer.target
    logger.info(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Classes: {len(set(y))} (0: Malignant, 1: Benign)")

    # Split the data into training and testing sets
    logger.info("Splitting data into train/test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"Training set size: {X_train.shape[0]} samples")
    logger.info(f"Testing set size: {X_test.shape[0]} samples")

    # Train an XGBoost classifier
    logger.info("Training XGBoost classifier...")
    model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    model.fit(X_train, y_train)
    logger.info("Model training completed")

    # Evaluate the model
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    logger.info(f"Training accuracy: {train_accuracy:.4f}")
    logger.info(f"Testing accuracy: {test_accuracy:.4f}")

    # Save the model to a file
    logger.info("Saving model to breast_cancer_model.pkl...")
    joblib.dump(model, 'breast_cancer_model.pkl')
    logger.info("Model saved successfully")
    
    logger.info("Model training pipeline completed successfully!")

