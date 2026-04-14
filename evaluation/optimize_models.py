"""
Model Optimization Suite
ONNX quantization + INT8 inference for 75%+ memory reduction
"""

import torch
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from typing import Tuple, Optional
import json
import time
import psutil

class ModelOptimizer:
    """
    Optimize transformer models for production deployment
    """
    
    def __init__(self, output_dir: str = "optimized_models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def get_model_memory(self, model) -> float:
        """Get model memory usage in MB"""
        param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    def quantize_dynamic(self, model_name: str, save_name: str) -> Tuple[bool, float, float]:
        """
        Apply dynamic INT8 quantization
        Returns: (success, original_mb, quantized_mb)
        """
        print(f"\n🔧 Quantizing {model_name}...")
        
        try:
            # Load original model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            model.eval()
            
            original_memory = self.get_model_memory(model)
            print(f"  Original size: {original_memory:.1f}MB")
            
            # Apply dynamic quantization
            quantized_model = torch.quantization.quantize_dynamic(
                model, 
                {torch.nn.Linear},  # Quantize linear layers
                dtype=torch.qint8
            )
            
            quantized_memory = self.get_model_memory(quantized_model)
            print(f"  Quantized size: {quantized_memory:.1f}MB")
            
            reduction = (1 - quantized_memory / original_memory) * 100
            print(f"  Memory reduction: {reduction:.1f}%")
            
            # Save quantized model
            save_path = self.output_dir / save_name
            torch.save(quantized_model.state_dict(), save_path / "pytorch_model.bin")
            tokenizer.save_pretrained(save_path)
            
            # Save config
            config = {
                "original_model": model_name,
                "quantization": "dynamic_int8",
                "original_mb": original_memory,
                "quantized_mb": quantized_memory,
                "reduction_percent": reduction
            }
            
            with open(save_path / "optimization_config.json", "w") as f:
                json.dump(config, f, indent=2)
            
            return True, original_memory, quantized_memory
            
        except Exception as e:
            print(f"  ❌ Quantization failed: {e}")
            return False, 0.0, 0.0
    
    def export_to_onnx(self, model_name: str, save_name: str) -> Tuple[bool, float]:
        """
        Export model to ONNX format for faster inference
        """
        print(f"\n📦 Exporting {model_name} to ONNX...")
        
        try:
            from optimum.onnxruntime import ORTModelForSequenceClassification
            
            # Export to ONNX
            save_path = self.output_dir / f"{save_name}_onnx"
            
            model = ORTModelForSequenceClassification.from_pretrained(
                model_name,
                export=True,
                provider="CPUExecutionProvider"
            )
            model.save_pretrained(save_path)
            
            print(f"  ✅ ONNX model saved to {save_path}")
            return True, 0.0
            
        except ImportError:
            print("  ⚠️ optimum[onnxruntime] not installed. Install with:")
            print("     pip install optimum[onnxruntime]")
            return False, 0.0
        except Exception as e:
            print(f"  ❌ ONNX export failed: {e}")
            return False, 0.0
    
    def benchmark_inference(self, model_name: str, num_runs: int = 100) -> Dict:
        """
        Benchmark inference speed
        """
        print(f"\n⏱️  Benchmarking {model_name}...")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            model.eval()
            
            test_text = "This is a test prompt for benchmarking inference speed."
            
            # Warmup
            for _ in range(10):
                inputs = tokenizer(test_text, return_tensors="pt", truncation=True)
                with torch.no_grad():
                    _ = model(**inputs)
            
            # Benchmark
            times = []
            for _ in range(num_runs):
                start = time.perf_counter()
                
                inputs = tokenizer(test_text, return_tensors="pt", truncation=True)
                with torch.no_grad():
                    _ = model(**inputs)
                
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)
            
            results = {
                "mean_ms": np.mean(times),
                "p50_ms": np.percentile(times, 50),
                "p95_ms": np.percentile(times, 95),
                "p99_ms": np.percentile(times, 99),
                "min_ms": np.min(times),
                "max_ms": np.max(times)
            }
            
            print(f"  Mean: {results['mean_ms']:.1f}ms")
            print(f"  P95:  {results['p95_ms']:.1f}ms")
            print(f"  P99:  {results['p99_ms']:.1f}ms")
            
            return results
            
        except Exception as e:
            print(f"  ❌ Benchmark failed: {e}")
            return {}
    
    def optimize_all_models(self):
        """
        Optimize all models in the gateway
        """
        print("=" * 70)
        print("🔧 Model Optimization Suite")
        print("=" * 70)
        
        models_to_optimize = [
            ("unitary/toxic-bert", "toxicity_quantized"),
            ("protectai/deberta-v3-base-prompt-injection", "injection_quantized"),
        ]
        
        results = {}
        total_original = 0
        total_quantized = 0
        
        for model_name, save_name in models_to_optimize:
            print(f"\n{'='*70}")
            print(f"Processing: {model_name}")
            print(f"{'='*70}")
            
            # Quantize
            success, orig_mb, quant_mb = self.quantize_dynamic(model_name, save_name)
            
            if success:
                total_original += orig_mb
                total_quantized += quant_mb
                
                # Benchmark
                speed_results = self.benchmark_inference(model_name)
                
                results[model_name] = {
                    "original_mb": orig_mb,
                    "quantized_mb": quant_mb,
                    "reduction_percent": (1 - quant_mb / orig_mb) * 100,
                    "speed": speed_results
                }
                
                # Try ONNX export
                self.export_to_onnx(model_name, save_name)
        
        # Summary
        if total_original > 0:
            overall_reduction = (1 - total_quantized / total_original) * 100
            
            print("\n" + "=" * 70)
            print("📊 OPTIMIZATION SUMMARY")
            print("=" * 70)
            print(f"Total original size:  {total_original:.1f}MB")
            print(f"Total quantized size: {total_quantized:.1f}MB")
            print(f"Overall reduction:    {overall_reduction:.1f}%")
            print(f"Memory saved:         {total_original - total_quantized:.1f}MB")
            
            # Save summary
            summary = {
                "total_original_mb": total_original,
                "total_quantized_mb": total_quantized,
                "overall_reduction_percent": overall_reduction,
                "models": results,
                "timestamp": time.time()
            }
            
            with open(self.output_dir / "optimization_summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            
            print(f"\n✅ Results saved to {self.output_dir}/optimization_summary.json")
            
            return overall_reduction
        
        return 0.0

def create_optimized_gateway_config():
    """
    Create configuration to use optimized models
    """
    config = """# Optimized Model Configuration
# Use these environment variables to load quantized models

# Original models
TOXICITY_MODEL=unitary/toxic-bert
INJECTION_MODEL=protectai/deberta-v3-base-prompt-injection

# Optimized models (after running optimize_models.py)
# TOXICITY_MODEL=./optimized_models/toxicity_quantized
# INJECTION_MODEL=./optimized_models/injection_quantized

# Enable ONNX Runtime (requires optimum library)
# USE_ONNX=true
"""
    
    with open("optimized_config.env", "w") as f:
        f.write(config)
    
    print("\n📝 Created optimized_config.env")

if __name__ == "__main__":
    optimizer = ModelOptimizer()
    reduction = optimizer.optimize_all_models()
    create_optimized_gateway_config()
    
    if reduction >= 75:
        print(f"\n🎉 Achieved {reduction:.1f}% memory reduction! Target met.")
    else:
        print(f"\n⚠️  Achieved {reduction:.1f}% reduction. Target: 75%+")
        print("   Consider aggressive quantization or model distillation.")
