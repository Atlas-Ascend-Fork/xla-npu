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
    
    from config import (
        NPU_PLATFORM_NAME,
        NPU_LIBRARY_NAME,
        ENV_PJRT_NAMES_AND_LIBRARY_PATHS,
        ENV_JAX_PLATFORMS,
    )
    
    LIBRARY_PATH = os.path.abspath(NPU_LIBRARY_NAME)
    os.environ[ENV_PJRT_NAMES_AND_LIBRARY_PATHS] = f"{NPU_PLATFORM_NAME}:{LIBRARY_PATH}"
    os.environ[ENV_JAX_PLATFORMS] = NPU_PLATFORM_NAME
    
    import jax
    # Do NOT reload jax - this would trigger recompilation
    
    _initialized = True


def get_model_and_params():
    """Get or create model and parameters (cached)."""
    global _model, _params
    
    if _model is None or _params is None:
        from test_utils import setup_model_and_params
        _model, _params = setup_model_and_params()
    
    return _model, _params


def run_npu_test(num_runs=1, input_data=None):
    """Run NPU inference test multiple times.
    
    Args:
        num_runs: Number of runs to execute
        input_data: Pre-generated input data. If None, generates new input.
    """
    initialize_npu()
    
    from test_utils import generate_input, run_inference_and_save
    from config import NPU_OUTPUT_FILE, NPU_ALL_OUTPUTS_FILE
    
    # Get cached model and params
    model, params = get_model_and_params()
    
    # Generate or use provided input data
    if input_data is None:
        x = generate_input()
    else:
        x = input_data
    
    # Run inference multiple times
    outputs = []
    
    for i in range(num_runs):
        print(f"[NPU] Running inference {i+1}/{num_runs}...", flush=True)
        output = run_inference_and_save(model, params, x, "NPU", NPU_OUTPUT_FILE)
        outputs.append(output.copy())
        print(f"[NPU] Run {i+1}/{num_runs} completed", flush=True)
    
    # Save all outputs for comparison
    all_outputs = np.stack(outputs, axis=0)
    np.savez(NPU_ALL_OUTPUTS_FILE, 
             input=x, 
             outputs=all_outputs,
             num_runs=num_runs)
    
    print(f"[NPU] All {num_runs} runs completed and saved", flush=True)
    return outputs, x


if __name__ == "__main__":
    # Get num_runs from environment
    num_runs = int(os.environ.get('NUM_RUNS', '1'))
    
    # Check if input data is provided via environment variable
    input_data = None
    input_file = os.environ.get('INPUT_DATA_FILE', None)
    if input_file and os.path.exists(input_file):
        data = np.load(input_file)
        input_data = data['input']
    
    outputs, x = run_npu_test(num_runs, input_data)
