import logging
import joblib
from xgboost import XGBClassifier
from data import load_data, split_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def fit_model(X_train, y_train, X_test, y_test):
    """
    Train an XGBoost classifier and save the model to a file.
    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training target values.
    """
    model = XGBClassifier(n_estimators=100, random_state=42, eval_metric="mlogloss")
    model.fit(X_train, y_train)
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    logger.info("Training accuracy: %.4f", train_accuracy)
    logger.info("Testing accuracy: %.4f", test_accuracy)
    logger.info("Saving model to wine_model.pkl...")
    joblib.dump(model, "../model/wine_model.pkl")
    logger.info("Model saved successfully")

if __name__ == "__main__":
    logger.info("Starting model training pipeline...")

    logger.info("Loading Wine dataset...")
    X, y = load_data()
    logger.info("Dataset loaded: %s samples, %s features", X.shape[0], X.shape[1])
    logger.info("Classes: %s", len(set(y)))

    logger.info("Splitting data into train/test sets (70/30)...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    logger.info("Training set size: %s samples", X_train.shape[0])
    logger.info("Testing set size: %s samples", X_test.shape[0])

    logger.info("Training XGBoost classifier...")
    fit_model(X_train, y_train, X_test, y_test)
    logger.info("Model training completed")

    logger.info("Training pipeline completed successfully")
