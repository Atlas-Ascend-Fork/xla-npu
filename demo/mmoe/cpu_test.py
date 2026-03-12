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


def run_cpu_test(num_runs=1, inputs=None):
    """Run CPU inference test multiple times.
    
    Args:
        num_runs: Number of runs to execute
        inputs: List of input data for each run. If None, generates new inputs.
    """
    from test_utils import generate_input, run_inference_and_save
    from config import CPU_OUTPUT_FILE, CPU_ALL_OUTPUTS_FILE
    
    # Get cached model and params
    model, params = get_model_and_params()
    
    # Run inference multiple times with different inputs
    outputs = []
    
    for i in range(num_runs):
        if inputs is not None and i < len(inputs):
            x = inputs[i]
        else:
            x = generate_input()
        
        print(f"[CPU] Running inference {i+1}/{num_runs}...", flush=True)
        output = run_inference_and_save(model, params, x, "CPU", CPU_OUTPUT_FILE)
        outputs.append(output.copy())
        print(f"[CPU] Run {i+1}/{num_runs} completed", flush=True)
    
    # Save all outputs for comparison
    all_outputs = np.stack(outputs, axis=0)
    all_inputs = np.stack([np.array(inp) for inp in (inputs if inputs else [generate_input()] * num_runs)], axis=0) if inputs else None
    save_dict = {
        'outputs': all_outputs,
        'num_runs': num_runs
    }
    if all_inputs is not None:
        save_dict['inputs'] = all_inputs
    np.savez(CPU_ALL_OUTPUTS_FILE, **save_dict)
    
    print(f"[CPU] All {num_runs} runs completed and saved", flush=True)
    return outputs, inputs


if __name__ == "__main__":
    # Get num_runs from environment
    num_runs = int(os.environ.get('NUM_RUNS', '1'))
    
    # Check if input data is provided via environment variable
    inputs = None
    input_file = os.environ.get('INPUT_DATA_FILE', None)
    if input_file and os.path.exists(input_file):
        data = np.load(input_file)
        if 'inputs' in data:
            inputs = list(data['inputs'])
            num_runs_in_file = int(data.get('num_runs', len(inputs)))
            num_runs = min(num_runs, num_runs_in_file)
        elif 'input' in data:
            inputs = [data['input']] * num_runs
    
    outputs, inputs = run_cpu_test(num_runs, inputs)
