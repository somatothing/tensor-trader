"""FastAPI serving layer for Tensor Trader model inference."""
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import pickle

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class PredictionRequest(BaseModel):
    """Request model for predictions."""
    features: List[List[float]] = Field(..., description="Feature matrix (n_samples, n_features)")
    model_type: Optional[str] = Field("decision_tree", description="Model type to use")


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predictions: List[int] = Field(..., description="Predicted class labels")
    probabilities: Optional[List[List[float]]] = Field(None, description="Class probabilities")
    model_type: str = Field(..., description="Model used for prediction")


class OHLCVRequest(BaseModel):
    """Request model for OHLCV data."""
    open: List[float]
    high: List[float]
    low: List[float]
    close: List[float]
    volume: List[float]
    symbol: str = Field("BTCUSDT", description="Trading symbol")
    timeframe: str = Field("1m", description="Timeframe")


class OHLCVResponse(BaseModel):
    """Response model for OHLCV predictions."""
    predictions: List[int] = Field(..., description="Predicted labels for each candle")
    probabilities: Optional[List[List[float]]] = Field(None, description="Class probabilities")
    features_used: int = Field(..., description="Number of features used")
    symbol: str = Field(..., description="Trading symbol")


class ModelInfo(BaseModel):
    """Model information response."""
    model_type: str
    input_dim: int
    feature_names: List[str]
    loaded: bool


class ModelManager:
    """Manages loaded models for inference."""
    
    def __init__(self, models_dir: str = "./models/checkpoints"):
        self.models_dir = Path(models_dir)
        self.models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict] = {}
    
    def load_model(self, model_path: str) -> bool:
        """Load a model from disk."""
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            self.models[model_path] = model
            logger.info(f"Model loaded from {model_path}")
            
            # Load metadata if available
            metadata_path = model_path.replace('.pkl', '_metadata.json')
            if Path(metadata_path).exists():
                with open(metadata_path) as f:
                    self.model_metadata[model_path] = json.load(f)
            
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def predict(self, model_path: str, X: np.ndarray) -> Dict[str, Any]:
        """Run prediction with loaded model."""
        if model_path not in self.models:
            raise ValueError(f"Model not loaded: {model_path}")
        
        model = self.models[model_path]
        predictions = model.predict(X)
        try:
            probabilities = model.predict_proba(X)
        except:
            probabilities = None
        
        return {
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist() if probabilities is not None else None
        }
    
    def list_models(self) -> List[str]:
        """List available models."""
        models = []
        if self.models_dir.exists():
            for f in self.models_dir.glob("*.pkl"):
                models.append(str(f))
        return models


# Create FastAPI app
app = FastAPI(
    title="Tensor Trader API",
    description="Real-time inference API for Tensor Trader models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model manager
model_manager = ModelManager()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Tensor Trader API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": len(model_manager.models)
    }


@app.get("/models")
async def list_models():
    """List available models."""
    return model_manager.list_models()


@app.post("/models/load")
async def load_model(model_path: str):
    """Load a model."""
    success = model_manager.load_model(model_path)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to load model")
    return {"status": "success", "model_path": model_path}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make predictions from features."""
    try:
        X = np.array(request.features)
        
        # Use first available model if none specified
        available_models = model_manager.list_models()
        if not available_models:
            raise HTTPException(status_code=400, detail="No models loaded")
        
        model_path = available_models[0]
        
        # Load model if not already loaded
        if model_path not in model_manager.models:
            model_manager.load_model(model_path)
        
        result = model_manager.predict(model_path, X)
        
        return PredictionResponse(
            predictions=result['predictions'],
            probabilities=result['probabilities'],
            model_type=request.model_type
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the FastAPI server."""
    uvicorn.run("tensor_trader.serving.api:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    run_server()
