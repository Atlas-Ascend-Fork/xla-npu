# -*- coding:utf-8 -*-
"""
MMOE Test Utilities - Common functions for CPU and NPU tests
"""

import numpy as np
import jax
import jax.numpy as jnp
from config import (
    CONFIG_BATCH_SIZE,
    CONFIG_INPUT_DIM,
    CONFIG_NUM_TASKS,
    COMPARISON_TOLERANCE,
    PRINT_SEPARATOR,
    OUTPUT_SAMPLE_SIZE,
)
from init_params import (
    setup_model_and_params,
)


# ============================================================================
# Input Generation
# ============================================================================

def generate_input(seed=None):
    """Generate random input data suitable for MMOE model.
    
    The input data follows a normal distribution with mean 0 and std 1,
    which is suitable for neural network inference after typical feature normalization.
    
    Args:
        seed: Random seed for reproducibility. If None, uses random seed.
    
    Returns:
        jnp.ndarray: Input tensor with shape (CONFIG_BATCH_SIZE, CONFIG_INPUT_DIM)
    """
    if seed is None:
        seed = int(np.random.randint(0, 10000))
    
    rng = np.random.RandomState(seed)
    x = rng.randn(CONFIG_BATCH_SIZE, CONFIG_INPUT_DIM).astype(np.float32)
    return jnp.array(x)


def print_input_stats(x, run_idx=None, device_name=None):
    """Print input data statistics.
    
    Args:
        x: Input tensor
        run_idx: Run index (optional, not used in output)
        device_name: Device name (optional, not used in output)
    """
    x_arr = np.array(x)
    
    print(f"\nInput:")
    print(f"  Shape: {x_arr.shape}")
    print(f"  Min: {x_arr.min():.6f}")
    print(f"  Max: {x_arr.max():.6f}")
    print(f"  Mean: {x_arr.mean():.6f}")
    print(f"  Std: {x_arr.std():.6f}")


def print_results(output, device_name, batch_size, num_tasks):
    """Print inference results.
    
    Args:
        output: Output tensor
        device_name: Device name ('CPU' or 'NPU')
        batch_size: Batch size
        num_tasks: Number of tasks
    """
    print(f"\n[{device_name}] Output:")
    print(f"[{device_name}]   Shape: {output.shape}")
    print(f"[{device_name}]   Min: {output.min():.6f}")
    print(f"[{device_name}]   Max: {output.max():.6f}")
    print(f"[{device_name}]   Mean: {output.mean():.6f}")
    print(f"[{device_name}]   Std: {output.std():.6f}")


# ============================================================================
# JIT Compilation and Inference
# ============================================================================

# Module-level JIT cache
_jit_cache = {}


def get_jitted_forward(model):
    """Get or create JIT-compiled forward function for the model.
    
    Args:
        model: MMOE model instance
        
    Returns:
        Callable: JIT-compiled forward function
    """
    model_id = id(model)
    if model_id not in _jit_cache:
        @jax.jit
        def jitted_forward(params, x):
            return model.apply(params, x, train=False)
        _jit_cache[model_id] = jitted_forward
    return _jit_cache[model_id]


def run_inference(model, params, x):
    """Run inference and return results.
    
    Args:
        model: MMOE model instance
        params: Model parameters
        x: Input tensor
        
    Returns:
        np.ndarray: Output as numpy array
    """
    jitted_forward = get_jitted_forward(model)
    output = jitted_forward(params, x)
    return np.array(output)


def run_inference_and_save(model, params, x, device_name, output_file):
    """Run inference and save results to file.
    
    Args:
        model: MMOE model instance
        params: Model parameters
        x: Input tensor
        device_name: Device name ('CPU' or 'NPU')
        output_file: Path to save output
        
    Returns:
        np.ndarray: Output as numpy array
    """
    output = run_inference(model, params, x)
    np.savez(output_file, output=output, input=x)
    return output


# ============================================================================
# Result Comparison and Printing
# ============================================================================

def compare_results(output_cpu, output_npu, x_cpu, x_npu):
    """Compare CPU and NPU results.
    
    Args:
        output_cpu: CPU output
        output_npu: NPU output
        x_cpu: CPU input
        x_npu: NPU input
        
    Returns:
        bool: True if results match within tolerance
    """
    if not np.allclose(x_cpu, x_npu):
        print("Warning: CPU and NPU inputs are different!")
    
    print(f"\n{PRINT_SEPARATOR}")
    print("--- Result Comparison (CPU vs NPU) ---")
    print(PRINT_SEPARATOR)
    
    diff = np.abs(output_cpu - output_npu)
    rel_diff = diff / (np.abs(output_cpu) + 1e-8)
    
    print(f"Absolute difference (first {OUTPUT_SAMPLE_SIZE} samples):")
    print(diff[:OUTPUT_SAMPLE_SIZE])
    print(f"\nDifference statistics:")
    print(f"  Max absolute difference: {diff.max():.6e}")
    print(f"  Mean absolute difference: {diff.mean():.6e}")
    print(f"  Max relative difference: {rel_diff.max():.6e}")
    print(f"  Mean relative difference: {rel_diff.mean():.6e}")
    
    is_match = np.allclose(output_cpu, output_npu, rtol=COMPARISON_TOLERANCE, atol=COMPARISON_TOLERANCE)
    print(f"\nResults match (tolerance={COMPARISON_TOLERANCE}): {is_match}")
    return is_match
