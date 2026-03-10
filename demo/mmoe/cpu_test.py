# -*- coding:utf-8 -*-
"""
MMOE CPU Test - Run inference on CPU multiple times
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Global variables for caching
_model = None
_params = None
_initialized = False


def get_model_and_params():
    """Get or create model and parameters (cached)."""
    global _model, _params
    
    if _model is None or _params is None:
        from test_utils import setup_model_and_params
        _model, _params = setup_model_and_params()
    
    return _model, _params


def run_cpu_test(num_runs=1):
    """Run CPU inference test multiple times.
    
    Args:
        num_runs: Number of runs to execute
    """
    from test_utils import generate_input, run_inference_and_save
    from config import CPU_OUTPUT_FILE, CPU_ALL_OUTPUTS_FILE
    
    # Get cached model and params
    model, params = get_model_and_params()
    
    # Generate input data (shared utility function)
    x = generate_input()
    
    # Run inference multiple times
    outputs = []
    
    for i in range(num_runs):
        print(f"[CPU] Running inference {i+1}/{num_runs}...", flush=True)
        output = run_inference_and_save(model, params, x, "CPU", CPU_OUTPUT_FILE)
        outputs.append(output.copy())
        print(f"[CPU] Run {i+1}/{num_runs} completed", flush=True)
    
    # Save all outputs for comparison
    all_outputs = np.stack(outputs, axis=0)
    np.savez(CPU_ALL_OUTPUTS_FILE, 
             input=x, 
             outputs=all_outputs,
             num_runs=num_runs)
    
    print(f"[CPU] All {num_runs} runs completed and saved", flush=True)
    return outputs, x


if __name__ == "__main__":
    # Get num_runs from environment
    num_runs = int(os.environ.get('NUM_RUNS', '1'))
    outputs, x = run_cpu_test(num_runs)
