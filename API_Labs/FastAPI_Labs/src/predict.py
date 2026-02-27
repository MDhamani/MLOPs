import logging
import joblib

logger = logging.getLogger(__name__)

def predict_data(X):
    """
    Predict the class labels for the input data.
    Args:
        X (numpy.ndarray): Input data for which predictions are to be made.
    Returns:
        y_pred (numpy.ndarray): Predicted class labels.
    """
    model = joblib.load("../model/wine_model.pkl")
    y_pred = model.predict(X)
    logger.info("Prediction completed for %s rows", len(X))
    return y_pred
