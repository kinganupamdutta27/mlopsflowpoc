from typing import List, Optional
from pydantic import BaseModel, Field

class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., description="Length of the sepal in centimeters")
    sepal_width: float = Field(..., description="Width of the sepal in centimeters")
    petal_length: float = Field(..., description="Length of the petal in centimeters")
    petal_width: float = Field(..., description="Width of the petal in centimeters")

    class Config:
        json_schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }


class PredictionRequest(BaseModel):
    features: List[IrisFeatures] = Field(..., description="List of iris flower features for prediction")
    model_version: Optional[str] = Field(None, description="Version of the model to be used for prediction")
    model_name: Optional[str] = Field("IrisClassifier_Sklearn", description="Name of the prediction model")

    class Config:
        json_schema_extra = {
            "example": {
                "features": [
                    {
                        "sepal_length": 5.1,
                        "sepal_width": 3.5,
                        "petal_length": 1.4,
                        "petal_width": 0.2
                    },
                    {
                        "sepal_length": 6.7,
                        "sepal_width": 3.1,
                        "petal_length": 4.7,
                        "petal_width": 1.5
                    }
                ],
                "model_version": "v1.0.0",
                "model_name": "IrisClassifier_Sklearn"
            }
        }


class PredictionResponse(BaseModel):
    predictions: List[int] = Field(..., description="List of predicted class labels for each input sample")
    probabilities: Optional[List[List[float]]] = Field(None, description="Predicted class probabilities for each input sample")
    model_version: str = Field(..., description="Version of the model that made the predictions")
    model_name: str = Field(..., description="Name of the model that made the predictions")

    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [0, 1],
                "probabilities": [
                    [0.95, 0.04, 0.01],
                    [0.02, 0.85, 0.13]
                ],
                "model_version": "v1.0.0",
                "model_name": "IrisClassifier_Sklearn"
            }
        }
