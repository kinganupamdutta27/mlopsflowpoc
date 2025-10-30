from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from app.schema import IrisFeatures, PredictionRequest, PredictionResponse
import mlflow.pyfunc
from app.utils import logger
from app.routers import mlops_router

from app.utils.utils import load_production_model, load_specific_version

app = FastAPI(title="ML Model Inference Service", version="1.0.0")

# Model registry
MODEL_REGISTRY = {}
MODEL_ALIASES = {}  # For A/B testing routing


# @app.on_event("startup")
# async def startup_event():
#     """Load production models on startup"""
#     try:
#         # Set MLflow tracking URI
#         mlflow.set_tracking_uri("http://localhost:5000")
        
#         # Load default production models
#         load_production_model("IrisClassifier_Sklearn")
#         load_production_model("IrisClassifier_PyTorch")
        
#     except Exception as e:
#         logger.error(f"Error during startup: {str(e)}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown lifecycle"""
    try:
        # --- Startup logic ---
        mlflow.set_tracking_uri("http://localhost:5000")
        load_production_model("IrisClassifier_Sklearn")
        load_production_model("IrisClassifier_PyTorch")
        logger.info("Models loaded successfully during startup.")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
    
    # yield control back to the app (runs while app is alive)
    yield

    # --- Shutdown logic (optional) ---
    logger.info("Application shutting down... cleanup if needed")

# create app with lifespan handler
app = FastAPI(lifespan=lifespan)

app.include_router(mlops_router)


