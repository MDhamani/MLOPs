import logging
import os
from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from sklearn.datasets import load_wine
from predict import predict_data
from data import WINE_FEATURES


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI()

APP_VERSION = os.getenv("APP_VERSION", "1.0.0")

class WineData(BaseModel):
    """
    Pydantic BaseModel representing wine chemical analysis measurements.

    Attributes:
        alcohol (float): Alcohol content.
        malic_acid (float): Malic acid level.
        ash (float): Ash content.
        alcalinity_of_ash (float): Alcalinity of ash.
        magnesium (float): Magnesium content.
        total_phenols (float): Total phenols.
        flavanoids (float): Flavanoids content.
        nonflavanoid_phenols (float): Nonflavanoid phenols.
        proanthocyanins (float): Proanthocyanins content.
        color_intensity (float): Color intensity.
        hue (float): Hue.
        od280_od315_of_diluted_wines (float): OD280/OD315 of diluted wines.
        proline (float): Proline content.
    """
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float

class WineResponse(BaseModel):
    response:int

class WineBatchRequest(BaseModel):
    items: list[WineData]

class WineBatchResponse(BaseModel):
    responses: list[int]



"""Modern web apps use a technique named routing. This helps the user remember the URLs. 
For instance, instead of having /booking.php they see /booking/. Instead of /account.asp?id=1234/ 
they’d see /account/1234/."""

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    """Concurrent (multiple tasks can run simultaneously)"""
    logger.info("Health check requested")
    return {"status": "healthy"}

@app.get("/features", status_code=status.HTTP_200_OK)
async def list_features():
    """Return the ordered list of model features."""
    logger.info("Feature list requested")
    return {"features": WINE_FEATURES}

@app.get("/version", status_code=status.HTTP_200_OK)
async def version():
    """Return the API version."""
    logger.info("Version requested")
    return {"version": APP_VERSION}

@app.get("/model-info", status_code=status.HTTP_200_OK)
async def model_info():
    """Return basic dataset/model metadata."""
    logger.info("Model info requested")
    wine = load_wine()
    return {
        "feature_count": len(WINE_FEATURES),
        "class_count": len(wine.target_names),
        "class_names": list(wine.target_names),
    }

@app.get("/classes", status_code=status.HTTP_200_OK)
async def classes():
    """Return the dataset class names."""
    logger.info("Class list requested")
    wine = load_wine()
    return {"classes": list(wine.target_names)}

@app.post("/predict", response_model=WineResponse)
async def predict_wine(wine_features: WineData):
    """
    Predict the wine cultivar based on provided features.
    This endpoint accepts wine measurements and returns the predicted class.
    Args:
        wine_features (WineData): A WineData object containing the 13 wine features.
    Returns:
        WineResponse: A response object containing:
            - response (int): The predicted wine class (0, 1, or 2)
    Raises:
        HTTPException: Returns a 500 status code with error details if prediction fails.
    Example:
        POST /predict
        {
            "alcohol": 13.2,
            "malic_acid": 1.8,
            "ash": 2.3,
            "alcalinity_of_ash": 15.5,
            "magnesium": 100.0,
            "total_phenols": 2.3,
            "flavanoids": 2.0,
            "nonflavanoid_phenols": 0.3,
            "proanthocyanins": 1.9,
            "color_intensity": 5.6,
            "hue": 1.0,
            "od280_od315_of_diluted_wines": 3.1,
            "proline": 750.0
        }
        Response:
        {
            "response": 0
        }
    """
    try:
        features = [[getattr(wine_features, name) for name in WINE_FEATURES]]

        logger.info("Prediction requested")

        prediction = predict_data(features)
        logger.info("Prediction completed: %s", int(prediction[0]))
        return WineResponse(response=int(prediction[0]))
    
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=WineBatchResponse)
async def predict_wine_batch(payload: WineBatchRequest):
    """
    Predict wine classes for a batch of records.
    Args:
        payload (WineBatchRequest): A list of wine feature records.
    Returns:
        WineBatchResponse: A list of predicted classes.
    """
    if not payload.items:
        raise HTTPException(status_code=400, detail="items must not be empty")

    try:
        features = [
            [getattr(item, name) for name in WINE_FEATURES]
            for item in payload.items
        ]

        logger.info("Batch prediction requested: %s rows", len(features))
        predictions = predict_data(features)
        responses = [int(value) for value in predictions]
        logger.info("Batch prediction completed")
        return WineBatchResponse(responses=responses)

    except Exception as e:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail=str(e))
    


    
