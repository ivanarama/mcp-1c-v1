from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, LoggingHandler
import logging
import json
import os
import uvicorn
from typing import List, Optional

app = FastAPI(title="Configurable Embeddings Service",
              description="API for generating embeddings with configurable models")

# Configure logging
# logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.DEBUG, handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

# Load configuration


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning("Config file not found, using default configuration")
        return {
            "model": {
                "name": "all-MiniLM-L6-v2",
                "trust_remote_code": False,
                "default_task": "retrieval.passage"
            },
            "models_info": {
                "all-MiniLM-L6-v2": {
                    "dimensions": 384,
                    "supports_task": False
                }
            }
        }


config = load_config()

# Define request model


class EmbeddingRequest(BaseModel):
    texts: List[str]
    task: str = config["model"]["default_task"]
    dimensions: Optional[int] = None


# Load the configured model
try:
    model_name = config["model"]["name"]
    trust_remote_code = config["model"]["trust_remote_code"]
    model = SentenceTransformer(
        model_name, trust_remote_code=trust_remote_code)

    # Get model info
    model_info = config["models_info"].get(model_name, {})
    model_dimensions = model_info.get("dimensions", 384)
    supports_task = model_info.get("supports_task", False)

    logger.info(
        f"Model {model_name} loaded successfully with {model_dimensions} dimensions")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise e


@app.post("/embed")
async def generate_embeddings(request: EmbeddingRequest):
    try:
        texts = request.texts
        task = request.task
        dimensions = request.dimensions or model_dimensions

        if not texts:
            raise HTTPException(status_code=400, detail="No texts provided")

        # Generate embeddings with different parameters based on model capabilities
        if supports_task:
            embeddings = model.encode(
                texts,
                task=task,
                prompt_name=task,
                normalize_embeddings=True,
                output_value="sentence_embedding",
                precision="float32",
                convert_to_numpy=True
            )
        else:
            # For models that don't support task parameter (like all-MiniLM-L6-v2)
            embeddings = model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True
            )

        # Convert embeddings to list for JSON serialization
        embeddings_list = embeddings.tolist()

        return {
            "embeddings": embeddings_list,
            "dimensions": len(embeddings_list[0]) if embeddings_list else 0,
            "model": model_name,
            "task": task if supports_task else "default",
            "input_texts": texts
        }

    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint for service monitoring"""
    return {
        "status": "healthy",
        "service": "embedding_service",
        "model": model_name
    }


@app.get("/model-info")
async def get_model_info():
    """Get information about the currently loaded model"""
    return {
        "model_name": model_name,
        "dimensions": model_dimensions,
        "supports_task": supports_task,
        "available_models": list(config["models_info"].keys())
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
