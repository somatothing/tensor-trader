"""FastAPI serving layer for Tensor Trader model inference."""
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import pickle
import os

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import inference engine
from ..inference.engine import InferenceEngine, TradingDecision, SignalDirection, MultiTimeframeInference
from ..features.pipeline import FeaturePipeline
from ..models.boosting.xgboost_model import MarketXGBoost
from ..models.tree.decision_tree import MarketDecisionTree
from ..models.gnn.spread_tensor import SpreadTensorModel

logger = logging.getLogger(__name__)


# Pydantic models for API requests/responses
class PredictionRequest(BaseModel):
    """Request model for predictions."""
    features: List[List[float]] = Field(..., description="Feature matrix (n_samples, n_features)")
    model_type: Optional[str] = Field("ensemble", description="Model type to use")


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predictions: List[int] = Field(..., description="Predicted class labels")
    probabilities: Optional[List[List[float]]] = Field(None, description="Class probabilities")
    model_type: str = Field(..., description="Model used for prediction")
    confidence: float = Field(..., description="Prediction confidence")


class OHLCVRequest(BaseModel):
    """Request model for OHLCV data."""
    open: List[float]
    high: List[float]
    low: List[float]
    close: List[float]
    volume: List[float]
    symbol: str = Field("BTCUSDT", description="Trading symbol")
    timeframe: str = Field("1m", description="Timeframe")


class TradingDecisionResponse(BaseModel):
    """Response model for trading decisions."""
    direction: str = Field(..., description="BUY, SELL, or HOLD")
    confidence: float = Field(..., description="Decision confidence (0-1)")
    entry_price: float = Field(..., description="Suggested entry price")
    stop_loss: float = Field(..., description="Stop loss level")
    take_profit: float = Field(..., description="Take profit level")
    position_size: float = Field(..., description="Suggested position size")
    risk_reward_ratio: float = Field(..., description="Risk/Reward ratio")
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(..., description="Timeframe")
    model_predictions: Dict[str, Any] = Field(..., description="Detailed model predictions")
    timestamp: str = Field(..., description="Decision timestamp")
    execute: bool = Field(False, description="Whether to execute the trade")


class ModelInfo(BaseModel):
    """Model information response."""
    model_type: str
    input_dim: int
    feature_names: List[str]
    loaded: bool


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: int
    inference_ready: bool
    version: str = "1.0.0"


class ModelManager:
    """Manages loaded models for inference."""
    
    def __init__(self, models_dir: str = "./models/checkpoints"):
        self.models_dir = Path(models_dir)
        self.models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict] = {}
        self.inference_engine: Optional[InferenceEngine] = None
        self.feature_pipeline = FeaturePipeline()
    
    def load_model(self, model_path: str, model_type: str = "auto") -> bool:
        """Load a model from disk."""
        try:
            path = Path(model_path)
            if not path.exists():
                # Try relative to models_dir
                path = self.models_dir / model_path
            
            if not path.exists():
                logger.error(f"Model file not found: {model_path}")
                return False
            
            # Determine model type from filename or extension
            if model_type == "auto":
                if "xgboost" in path.name.lower():
                    model_type = "xgboost"
                elif "tree" in path.name.lower() or "decision" in path.name.lower():
                    model_type = "tree"
                elif "tensor" in path.name.lower() or "spread" in path.name.lower():
                    model_type = "spread_tensor"
                else:
                    model_type = "unknown"
            
            # Load based on type
            if model_type == "xgboost":
                model = MarketXGBoost()
                model.load(str(path))
            elif model_type == "tree":
                model = MarketDecisionTree()
                model.load(str(path))
            elif model_type == "spread_tensor":
                model = SpreadTensorModel(input_dim=1)
                model.load(str(path))
            else:
                # Generic pickle load
                with open(path, 'rb') as f:
                    model = pickle.load(f)
            
            self.models[model_type] = model
            logger.info(f"Model loaded from {path} as {model_type}")
            
            # Load metadata if available
            metadata_path = str(path).replace('.pkl', '_metadata.json').replace('.json', '_metadata.json')
            if Path(metadata_path).exists():
                with open(metadata_path) as f:
                    self.model_metadata[model_type] = json.load(f)
            
            # Update inference engine
            self._update_inference_engine()
            
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def _update_inference_engine(self):
        """Update inference engine with loaded models."""
        if self.models:
            self.inference_engine = InferenceEngine(
                models=self.models,
                feature_pipeline=self.feature_pipeline
            )
            logger.info(f"Inference engine updated with {len(self.models)} models")
    
    def predict(self, X: np.ndarray, model_type: Optional[str] = None) -> Dict[str, Any]:
        """Run prediction with loaded model."""
        if not self.models:
            raise ValueError("No models loaded")
        
        if model_type and model_type in self.models:
            model = self.models[model_type]
            predictions = model.predict(X)
            try:
                probabilities = model.predict_proba(X)
            except:
                probabilities = None
            
            return {
                'predictions': predictions.tolist(),
                'probabilities': probabilities.tolist() if probabilities is not None else None,
                'model_type': model_type
            }
        
        # Use inference engine for ensemble prediction
        if self.inference_engine:
            direction, confidence, details = self.inference_engine.ensemble_predict(X)
            return {
                'predictions': [direction.value],
                'probabilities': details.get('probabilities', {}),
                'confidence': confidence,
                'model_type': 'ensemble',
                'details': details
            }
        
        raise ValueError("No inference engine available")
    
    def make_trading_decision(self, ohlcv_data: Dict) -> Optional[TradingDecision]:
        """Make a trading decision from OHLCV data."""
        if not self.inference_engine:
            raise ValueError("Inference engine not initialized")
        
        import pandas as pd
        df = pd.DataFrame({
            'open': ohlcv_data['open'],
            'high': ohlcv_data['high'],
            'low': ohlcv_data['low'],
            'close': ohlcv_data['close'],
            'volume': ohlcv_data['volume'],
        })
        
        symbol = ohlcv_data.get('symbol', 'BTCUSDT')
        timeframe = ohlcv_data.get('timeframe', '1m')
        balance = ohlcv_data.get('account_balance', 10000.0)
        
        return self.inference_engine.make_decision(df, symbol, timeframe, balance)
    
    def list_models(self) -> List[str]:
        """List available models."""
        return list(self.models.keys())
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            'loaded_models': self.list_models(),
            'inference_ready': self.inference_engine is not None,
            'feature_pipeline_ready': True,
            'model_metadata': self.model_metadata
        }


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
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "decide": "/decide",
            "models": "/models"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        models_loaded=len(model_manager.models),
        inference_ready=model_manager.inference_engine is not None,
        version="1.0.0"
    )


@app.get("/models")
async def list_models():
    """List available models."""
    return model_manager.get_model_info()


@app.post("/models/load")
async def load_model(model_path: str, model_type: str = "auto"):
    """Load a model."""
    success = model_manager.load_model(model_path, model_type)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to load model")
    return {"status": "success", "model_path": model_path, "model_type": model_type}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make predictions from features."""
    try:
        X = np.array(request.features)
        
        if not model_manager.models:
            raise HTTPException(status_code=400, detail="No models loaded")
        
        result = model_manager.predict(X, request.model_type if request.model_type != "ensemble" else None)
        
        return PredictionResponse(
            predictions=result['predictions'],
            probabilities=result.get('probabilities'),
            model_type=result.get('model_type', 'unknown'),
            confidence=result.get('confidence', 0.5)
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/decide", response_model=TradingDecisionResponse)
async def make_decision(request: OHLCVRequest):
    """Make a trading decision from OHLCV data."""
    try:
        if not model_manager.inference_engine:
            raise HTTPException(status_code=400, detail="Inference engine not ready. Load models first.")
        
        ohlcv_data = {
            'open': request.open,
            'high': request.high,
            'low': request.low,
            'close': request.close,
            'volume': request.volume,
            'symbol': request.symbol,
            'timeframe': request.timeframe,
        }
        
        decision = model_manager.make_trading_decision(ohlcv_data)
        
        if decision is None:
            return TradingDecisionResponse(
                direction="HOLD",
                confidence=0.0,
                entry_price=request.close[-1] if request.close else 0,
                stop_loss=0,
                take_profit=0,
                position_size=0,
                risk_reward_ratio=0,
                symbol=request.symbol,
                timeframe=request.timeframe,
                model_predictions={},
                timestamp=str(pd.Timestamp.now()),
                execute=False
            )
        
        return TradingDecisionResponse(
            direction=decision.direction.name,
            confidence=decision.confidence,
            entry_price=decision.entry_price,
            stop_loss=decision.stop_loss,
            take_profit=decision.take_profit,
            position_size=decision.position_size,
            risk_reward_ratio=decision.risk_reward_ratio,
            symbol=decision.symbol,
            timeframe=decision.timeframe,
            model_predictions=decision.model_predictions,
            timestamp=decision.timestamp.isoformat(),
            execute=decision.confidence > 0.7
        )
        
    except Exception as e:
        logger.error(f"Decision error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """Get model and inference metrics."""
    if model_manager.inference_engine:
        stats = model_manager.inference_engine.get_decision_stats()
    else:
        stats = {"error": "Inference engine not initialized"}
    
    return {
        "models": model_manager.get_model_info(),
        "decision_stats": stats
    }


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the FastAPI server."""
    uvicorn.run("tensor_trader.serving.api:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    run_server()
