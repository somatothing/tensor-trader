"""ONNX export functionality for Tensor Trader models."""
import numpy as np
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import json

logger = logging.getLogger(__name__)

# Try to import ONNX libraries
try:
    import onnx
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("skl2onnx not available, ONNX export disabled")

# Try to import onnxmltools for XGBoost
try:
    import onnxmltools
    from onnxmltools.convert import convert_xgboost as onnxmltools_convert_xgboost
    ONNXMLTOOLS_AVAILABLE = True
except ImportError:
    ONNXMLTOOLS_AVAILABLE = False
    logger.warning("onnxmltools not available, XGBoost ONNX export disabled")

try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False
    logger.warning("onnxruntime not available, ONNX inference disabled")


class ONNXExporter:
    """Export models to ONNX format."""
    
    def __init__(self, output_dir: str = "./models/onnx"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def export_sklearn_model(self, 
                             model: Any, 
                             model_name: str,
                             input_dim: int,
                             opset_version: int = 11) -> Optional[str]:
        """
        Export a scikit-learn compatible model to ONNX.
        
        Args:
            model: Trained sklearn model (XGBoost, DecisionTree, etc.)
            model_name: Name for the output file
            input_dim: Number of input features
            opset_version: ONNX opset version
        
        Returns:
            Path to exported ONNX file or None if failed
        """
        if not ONNX_AVAILABLE:
            logger.error("skl2onnx not available, cannot export")
            return None
        
        try:
            # Define input type
            initial_type = [('float_input', FloatTensorType([None, input_dim]))]
            
            # Convert model
            onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=opset_version)
            
            # Save model
            output_path = self.output_dir / f"{model_name}.onnx"
            with open(output_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            
            logger.info(f"Model exported to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            return None
    
    def export_xgboost(self, 
                       model: Any, 
                       model_name: str,
                       input_dim: int) -> Optional[str]:
        """Export XGBoost model to ONNX using onnxmltools."""
        if not ONNXMLTOOLS_AVAILABLE:
            logger.error("onnxmltools not available, cannot export XGBoost to ONNX")
            return None
        
        try:
            from onnxmltools.convert.common.data_types import FloatTensorType
            
            # Define input type
            initial_type = [('float_input', FloatTensorType([None, input_dim]))]
            
            # Convert model using onnxmltools
            onnx_model = onnxmltools_convert_xgboost(model, initial_types=initial_type)
            
            # Save model
            output_path = self.output_dir / f"{model_name}.onnx"
            with open(output_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            
            logger.info(f"XGBoost model exported to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"XGBoost ONNX export failed: {e}")
            return None
    
    def export_decision_tree(self,
                            model: Any,
                            model_name: str, 
                            input_dim: int) -> Optional[str]:
        """Export Decision Tree to ONNX."""
        return self.export_sklearn_model(model, model_name, input_dim)
    
    def verify_onnx_model(self, onnx_path: str) -> bool:
        """Verify an ONNX model is valid."""
        if not ONNX_AVAILABLE:
            return False
        
        try:
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            logger.info(f"ONNX model verified: {onnx_path}")
            return True
        except Exception as e:
            logger.error(f"ONNX verification failed: {e}")
            return False
    
    def get_model_info(self, onnx_path: str) -> Dict[str, Any]:
        """Get information about an ONNX model."""
        if not ONNX_AVAILABLE:
            return {"error": "onnx not available"}
        
        try:
            onnx_model = onnx.load(onnx_path)
            
            # Get input/output info
            inputs = []
            for input in onnx_model.graph.input:
                shape = [dim.dim_value if dim.dim_value else dim.dim_param 
                        for dim in input.type.tensor_type.shape.dim]
                inputs.append({
                    'name': input.name,
                    'shape': shape
                })
            
            outputs = []
            for output in onnx_model.graph.output:
                shape = [dim.dim_value if dim.dim_value else dim.dim_param 
                        for dim in output.type.tensor_type.shape.dim]
                outputs.append({
                    'name': output.name,
                    'shape': shape
                })
            
            return {
                'valid': True,
                'inputs': inputs,
                'outputs': outputs,
                'opset_version': onnx_model.opset_import[0].version if onnx_model.opset_import else None,
                'producer': onnx_model.producer_name,
            }
        except Exception as e:
            return {
                'valid': False,
                'error': str(e)
            }


class ONNXInference:
    """Inference using ONNX Runtime."""
    
    def __init__(self, onnx_path: str):
        """
        Initialize ONNX inference.
        
        Args:
            onnx_path: Path to ONNX model file
        """
        self.onnx_path = onnx_path
        self.session = None
        self.input_name = None
        self.output_names = None
        
        if ONNXRUNTIME_AVAILABLE:
            self._load_session()
        else:
            logger.error("onnxruntime not available")
    
    def _load_session(self):
        """Load ONNX Runtime session."""
        try:
            # Configure session for low memory usage
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 1
            sess_options.inter_op_num_threads = 1
            # Use DISABLE_ALL for compatibility with different onnxruntime versions
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            
            self.session = ort.InferenceSession(
                self.onnx_path, 
                sess_options,
                providers=['CPUExecutionProvider']
            )
            
            # Get input/output names
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]
            
            logger.info(f"ONNX session loaded: {self.onnx_path}")
            
        except Exception as e:
            logger.error(f"Failed to load ONNX session: {e}")
            self.session = None
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Run inference on input data.
        
        Args:
            X: Input features (n_samples, n_features)
        
        Returns:
            Predictions
        """
        if self.session is None:
            raise RuntimeError("ONNX session not loaded")
        
        # Ensure float32
        X = X.astype(np.float32)
        
        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: X})
        
        # Return first output (predictions)
        return outputs[0]
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Input features
        
        Returns:
            Class probabilities
        """
        if self.session is None:
            raise RuntimeError("ONNX session not loaded")
        
        X = X.astype(np.float32)
        outputs = self.session.run(self.output_names, {self.input_name: X})
        
        # Return second output if available (probabilities), else first
        if len(outputs) > 1:
            return outputs[1]
        return outputs[0]
    
    def benchmark(self, X: np.ndarray, n_runs: int = 100) -> Dict[str, float]:
        """Benchmark inference speed."""
        import time
        
        if self.session is None:
            return {"error": "Session not loaded"}
        
        X = X.astype(np.float32)
        
        # Warmup
        for _ in range(10):
            self.predict(X[:1])
        
        # Benchmark
        times = []
        for _ in range(n_runs):
            start = time.time()
            self.predict(X)
            times.append(time.time() - start)
        
        return {
            'mean_ms': np.mean(times) * 1000,
            'std_ms': np.std(times) * 1000,
            'min_ms': np.min(times) * 1000,
            'max_ms': np.max(times) * 1000,
            'n_runs': n_runs
        }


def export_all_models(models: Dict[str, Any], 
                     input_dim: int,
                     output_dir: str = "./models/onnx") -> Dict[str, str]:
    """
    Export all models to ONNX format.
    
    Args:
        models: Dictionary of trained models
        input_dim: Number of input features
        output_dir: Output directory
    
    Returns:
        Dictionary mapping model names to ONNX paths
    """
    exporter = ONNXExporter(output_dir)
    exported = {}
    
    for name, model in models.items():
        if name == 'xgboost':
            path = exporter.export_xgboost(model.model, 'xgboost', input_dim)
        elif name == 'tree':
            path = exporter.export_decision_tree(model.model, 'decision_tree', input_dim)
        else:
            path = exporter.export_sklearn_model(model, name, input_dim)
        
        if path:
            exported[name] = path
            exporter.verify_onnx_model(path)
    
    return exported


if __name__ == '__main__':
    # Test ONNX export
    print("Testing ONNX export...")
    
    # Create dummy model
    from sklearn.tree import DecisionTreeClassifier
    
    X = np.random.randn(100, 10)
    y = np.random.choice([0, 1], 100)
    
    model = DecisionTreeClassifier(max_depth=5)
    model.fit(X, y)
    
    # Export
    exporter = ONNXExporter()
    onnx_path = exporter.export_sklearn_model(model, 'test_tree', 10)
    
    if onnx_path:
        print(f"Exported to: {onnx_path}")
        
        # Test inference
        inference = ONNXInference(onnx_path)
        pred = inference.predict(X[:5])
        print(f"Predictions: {pred}")
        
        # Benchmark
        bench = inference.benchmark(X)
        print(f"Benchmark: {bench['mean_ms']:.2f}ms avg")
    else:
        print("Export failed")
