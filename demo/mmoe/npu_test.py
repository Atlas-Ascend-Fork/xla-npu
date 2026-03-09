# -*- coding:utf-8 -*-
"""
MMOE NPU Test - Run inference on NPU multiple times
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Global variables for caching
_model = None
_params = None
_initialized = False


def initialize_npu():
    """Initialize NPU environment only once."""
    global _initialized
    
    if _initialized:
        return
    
    PLATFORM_NAME = 'npu'
    LIBRARY_PATH = os.path.abspath("pjrt_c_api_npu_plugin.so")
    os.environ['PJRT_NAMES_AND_LIBRARY_PATHS'] = f"{PLATFORM_NAME}:{LIBRARY_PATH}"
    os.environ['JAX_PLATFORMS'] = PLATFORM_NAME
    
    import jax
    # Do NOT reload jax - this would trigger recompilation
    
    _initialized = True


def get_model_and_params():
    """Get or create model and parameters (cached)."""
    global _model, _params
    
    if _model is None or _params is None:
        from model import setup_model_and_data
        _model, _params, _ = setup_model_and_data()
    
    return _model, _params


def run_npu_test(num_runs=1, test_idx=1):
    """Run NPU inference test multiple times.
    
    Args:
        num_runs: Number of runs to execute
        test_idx: Test index (1-based), used to generate different inputs
    """
    initialize_npu()
    
    import jax.numpy as jnp
    from model import (
        CONFIG_BATCH_SIZE,
        CONFIG_INPUT_DIM,
    )
    
    # Get cached model and params
    model, params = get_model_and_params()
    
    # Prepare input data with test-specific sequence (same as CPU)
    total_elements = CONFIG_BATCH_SIZE * CONFIG_INPUT_DIM
    start_value = 1  # Start from 1
    x_values = np.arange(start_value, start_value + total_elements, dtype=np.float32)
    x = jnp.array(x_values.reshape((CONFIG_BATCH_SIZE, CONFIG_INPUT_DIM)))
    
    # Run inference multiple times
    from model import run_inference_and_save
    outputs = []
    
    for i in range(num_runs):
        print(f"[NPU] Running inference {i+1}/{num_runs}...", flush=True)
        output_file = f"/tmp/mmoe_npu_output.npz"
        output = run_inference_and_save(model, params, x, "NPU", output_file)
        outputs.append(output.copy())
        print(f"[NPU] Run {i+1}/{num_runs} completed", flush=True)
    
    # Save all outputs for comparison
    all_outputs = np.stack(outputs, axis=0)
    np.savez("/tmp/mmoe_npu_all_outputs.npz", 
             input=x, 
             outputs=all_outputs,
             num_runs=num_runs)
    
    print(f"[NPU] All {num_runs} runs completed and saved", flush=True)
    return outputs, x


if __name__ == "__main__":
    # Get num_runs and test_idx from environment
    num_runs = int(os.environ.get('NUM_RUNS', '1'))
    test_idx = int(os.environ.get('TEST_IDX', '1'))
    outputs, x = run_npu_test(num_runs, test_idx)
