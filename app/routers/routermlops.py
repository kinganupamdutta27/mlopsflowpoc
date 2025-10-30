from fastapi import APIRouter, HTTPException
import mlflow
import numpy as np

from app.schema.schema import PredictionRequest, PredictionResponse
from app.utils import logger
from app.utils.utils import MODEL_REGISTRY, load_production_model, load_specific_version

router = APIRouter(prefix="/mlops", tags=["MLOps Endpoints"])

@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make predictions with specified model version"""
    try:
        logger.info("Received prediction request")

        # Determine which model to use
        if request.model_version:
            logger.info(f"Model version specified: {request.model_version}")
            model_info = load_specific_version(request.model_name, request.model_version)
        else:
            logger.info(f"No model version specified, loading production model: {request.model_name}")
            model_info = MODEL_REGISTRY.get(f"{request.model_name}_production")
            if not model_info:
                logger.info(f"Production model not found in registry, loading from MLflow: {request.model_name}")
                load_production_model(request.model_name)
                model_info = MODEL_REGISTRY.get(f"{request.model_name}_production")
        
        if not model_info:
            logger.error("Model information could not be retrieved")
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Prepare features
        logger.info("Preparing input features for prediction")
        features_array = np.array([[f.sepal_length, f.sepal_width, f.petal_length, f.petal_width] 
                                   for f in request.features])
        logger.info(f"Features array shape: {features_array.shape}")

        # Make prediction
        logger.info("Running model prediction")
        model = model_info["model"]
        predictions = model.predict(features_array)
        logger.info(f"Predictions generated: {predictions.tolist()}")

        # Get probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            logger.info("Model supports probability predictions, calculating probabilities")
            probabilities = model.predict_proba(features_array).tolist()
            logger.info(f"Probabilities generated: {probabilities}")

        logger.info("Constructing PredictionResponse object for output")
        response = PredictionResponse(
            predictions=predictions.tolist(),
            probabilities=probabilities,
            model_version=model_info["version"],
            model_name=model_info["name"]
        )
        logger.info("Prediction completed successfully")
        return response

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/deploy/{model_name}/version/{version}")
async def deploy_model(model_name: str, version: str, stage: str = "Production"):
    """Deploy a specific model version to a stage"""
    try:
        logger.info(f"Starting deployment for model: {model_name}, version: {version}, stage: {stage}")
        
        client = mlflow.tracking.MlflowClient()
        logger.info("Mlflow client initialized successfully")

        # Transition model to specified stage
        logger.info(f"Transitioning model {model_name} version {version} to stage {stage}")
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=True
        )
        logger.info(f"Model {model_name} version {version} transitioned to {stage} successfully")

        # Load the newly deployed model
        logger.info(f"Loading production model: {model_name}")
        load_production_model(model_name)
        logger.info(f"Model {model_name} loaded successfully into production registry")

        logger.info(f"Deployment completed successfully for model {model_name} version {version}")
        return {"status": "success", "message": f"Model {model_name} version {version} deployed to {stage}"}
        
    except Exception as e:
        logger.error(f"Deployment failed for model {model_name} version {version}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_name}/versions")
async def get_model_versions(model_name: str):
    """Get all versions of a model"""
    try:
        logger.info(f"Fetching all versions for model: {model_name}")
        client = mlflow.tracking.MlflowClient()
        logger.debug("Initialized MLflow client successfully.")
        
        versions = client.search_model_versions(f"name='{model_name}'")
        logger.debug(f"Found {len(versions)} versions for model: {model_name}")
        
        version_info = []
        for v in versions:
            logger.debug(f"Processing version: {v.version}, stage: {v.current_stage}")
            version_info.append({
                "version": v.version,
                "current_stage": v.current_stage,
                "run_id": v.run_id,
                "created_at": v.creation_timestamp,
                "last_updated": v.last_updated_timestamp
            })
        
        logger.info(f"Successfully retrieved version info for model: {model_name}")
        return {"model_name": model_name, "versions": version_info}
        
    except Exception as e:
        logger.error(f"Error fetching versions for {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# A/B Testing Endpoints
@router.post("/ab-test/predict")
async def ab_test_predict(request: PredictionRequest, group: str = "A"):
    """A/B testing endpoint that routes to different model versions"""
    try:
        logger.info(f"A/B test prediction request received for group: {group}")
        logger.debug(f"Request details: {request}")
        
        # Define A/B test groups
        ab_test_config = {
            "A": {"model_name": "IrisClassifier_Sklearn", "version": "Production"},
            "B": {"model_name": "IrisClassifier_PyTorch", "version": "Production"}
        }
        
        config = ab_test_config.get(group, ab_test_config["A"])
        logger.info(f"Selected config: {config}")
        
        # Load appropriate model
        model_info = load_specific_version(config["model_name"], config["version"])
        logger.info(f"Loaded model {model_info['name']} version {model_info['version']} for group {group}")
        
        # Prepare features and predict
        features_array = np.array([[f.sepal_length, f.sepal_width, f.petal_length, f.petal_width] 
                                 for f in request.features])
        logger.debug(f"Prepared features array with shape: {features_array.shape}")
        
        model = model_info["model"]
        predictions = model.predict(features_array)
        logger.info(f"Prediction successful for group {group}")
        
        return {
            "predictions": predictions.tolist(),
            "model_version": model_info["version"],
            "model_name": model_info["name"],
            "test_group": group
        }
        
    except Exception as e:
        logger.error(f"A/B test prediction error for group {group}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    logger.info("Health check endpoint called.")
    status = {"status": "healthy", "loaded_models": list(MODEL_REGISTRY.keys())}
    logger.debug(f"Health status: {status}")
    return status
