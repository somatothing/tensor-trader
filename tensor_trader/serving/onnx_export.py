"""ONNX export functionality for Tensor Trader models."""
import numpy as np
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
import json

logger = logging.getLogger(__name__)

# Check for ONNX support
try:
    import skl2onnx
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    SKL2ONNX_AVAILABLE = True
except ImportError:
    SKL2ONNX_AVAILABLE = False
    logger.warning("skl2onnx not available, ONNX export disabled")

try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("onnx not available")

try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False
    logger.warning("onnxruntime not available")


class ONNXExporter:
    """Export sklearn models to ONNX format."""
    
    def __init__(self, model: Any, feature_names: Optional[List[str]] = None):
        self.model = model
        self.feature_names = feature_names or []
        self.input_dim = None
        
    def export(self, output_path: str, input_dim: Optional[int] = None):
        """Export model to ONNX format.
        
        Args:
            output_path: Path to save ONNX model
            input_dim: Input feature dimension (inferred if not provided)
        """
        if not SKL2ONNX_AVAILABLE:
            raise RuntimeError("skl2onnx not available. Install with: pip install skl2onnx")
        
        if input_dim is None:
            # Try to infer from model
            if hasattr(self.model, 'n_features_in_'):
                input_dim = self.model.n_features_in_
            else:
                raise ValueError("input_dim must be provided or model must have n_features_in_ attribute")
        
        self.input_dim = input_dim
        
        # Define input type
        initial_type = [('float_input', FloatTensorType([None, input_dim]))]
        
        # Convert model
        onnx_model = convert_sklearn(self.model, initial_types=initial_type)
        
        # Save model
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        
        logger.info(f"Model exported to {output_path}")
        
        # Save metadata
        metadata_path = output_path.replace('.onnx', '_metadata.json')
        metadata = {
            'input_dim': input_dim,
            'feature_names': self.feature_names,
            'model_type': type(self.model).__name__,
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return output_path
    
    def verify_export(self, onnx_path: str, sample_input: Optional[np.ndarray] = None) -> bool:
        """Verify ONNX export by loading and running inference.
        
        Args:
            onnx_path: Path to ONNX model
            sample_input: Optional sample input for testing
        
        Returns:
            True if verification successful
        """
        if not ONNXRUNTIME_AVAILABLE:
            logger.warning("onnxruntime not available, skipping verification")
            return False
        
        try:
            # Load ONNX model
            session = ort.InferenceSession(onnx_path)
            
            # Get input info
            input_name = session.get_inputs()[0].name
            input_shape = session.get_inputs()[0].shape
            
            logger.info(f"ONNX input: {input_name}, shape: {input_shape}")
            
            # Create sample input if not provided
            if sample_input is None:
                if self.input_dim:
                    sample_input = np.random.randn(1, self.input_dim).astype(np.float32)
                else:
                    sample_input = np.random.randn(1, 10).astype(np.float32)
            
            # Run inference
            outputs = session.run(None, {input_name: sample_input})
            
            logger.info(f"ONNX output shapes: {[o.shape for o in outputs]}")
            logger.info("ONNX verification: SUCCESS")
            
            return True
            
        except Exception as e:
            logger.error(f"ONNX verification failed: {e}")
            return False


class ONNXInference:
    """ONNX model inference wrapper."""
    
    def __init__(self, model_path: str):
        if not ONNXRUNTIME_AVAILABLE:
            raise RuntimeError("onnxruntime not available")
        
        self.model_path = model_path
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        
        # Load metadata if available
        metadata_path = model_path.replace('.onnx', '_metadata.json')
        self.metadata = {}
        if Path(metadata_path).exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)
        
        logger.info(f"ONNX model loaded from {model_path}")
        logger.info(f"Input: {self.input_name}, shape: {self.session.get_inputs()[0].shape}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run inference.
        
        Args:
            X: Input features (n_samples, n_features)
        
        Returns:
            Predictions
        """
        # Ensure float32
        X = X.astype(np.float32)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: X})
        
        # Return first output (predictions)
        return outputs[0]
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities.
        
        Args:
            X: Input features
        
        Returns:
            Class probabilities
        """
        X = X.astype(np.float32)
        outputs = self.session.run(None, {self.input_name: X})
        
        # Usually second output is probabilities for classifiers
        if len(outputs) > 1:
            return outputs[1]
        return outputs[0]
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get model metadata."""
        return {
            'input_shape': self.session.get_inputs()[0].shape,
            'output_shapes': [o.shape for o in self.session.get_outputs()],
            'metadata': self.metadata,
        }


def export_model_to_onnx(model: Any, output_path: str, 
                         feature_names: Optional[List[str]] = None,
                         input_dim: Optional[int] = None) -> str:
    """Convenience function to export a model to ONNX.
    
    Args:
        model: Sklearn model to export
        output_path: Path to save ONNX model
        feature_names: Optional list of feature names
        input_dim: Input dimension (inferred if not provided)
    
    Returns:
        Path to exported model
    """
    exporter = ONNXExporter(model, feature_names)
    return exporter.export(output_path, input_dim)


if __name__ == '__main__':
    # Test ONNX export
    print("Testing ONNX export...")
    
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.datasets import make_classification
    
    # Create sample model
    X, y = make_classification(n_samples=100, n_features=10, n_classes=3, random_state=42)
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X, y)
    
    # Export to ONNX
    if SKL2ONNX_AVAILABLE:
        output_path = "/tmp/test_model.onnx"
        exporter = ONNXExporter(model, feature_names=[f"f{i}" for i in range(10)])
        exporter.export(output_path, input_dim=10)
        
        # Verify
        success = exporter.verify_export(output_path, X[:1].astype(np.float32))
        print(f"Export verification: {'SUCCESS' if success else 'FAILED'}")
        
        # Test inference
        if ONNXRUNTIME_AVAILABLE:
            inference = ONNXInference(output_path)
            predictions = inference.predict(X[:5])
            print(f"Sample predictions: {predictions}")
    else:
        print("skl2onnx not available, skipping test")
