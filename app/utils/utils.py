import mlflow
from app.utils import logger
# Model registry
MODEL_REGISTRY = {}
MODEL_ALIASES = {}  # For A/B testing routing

def load_production_model(model_name: str):
    """Load the production version of a model"""
    try:
        model_uri = f"models:/{model_name}/Production"
        model = mlflow.pyfunc.load_model(model_uri)
        MODEL_REGISTRY[f"{model_name}_production"] = {
            "model": model,
            "version": "Production",
            "name": model_name
        }
        logger.info(f"Loaded production model: {model_name}")
    except Exception as e:
        logger.error(f"Error loading production model {model_name}: {str(e)}")
        raise

def load_specific_version(model_name: str, version: str):
    """Load a specific model version"""
    try:
        cache_key = f"{model_name}_v{version}"
        if cache_key not in MODEL_REGISTRY:
            model_uri = f"models:/{model_name}/{version}"
            model = mlflow.pyfunc.load_model(model_uri)
            MODEL_REGISTRY[cache_key] = {
                "model": model,
                "version": version,
                "name": model_name
            }
        return MODEL_REGISTRY[cache_key]
    except Exception as e:
        logger.error(f"Error loading model {model_name} version {version}: {str(e)}")
        raise