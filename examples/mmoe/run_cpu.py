#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the &quot;License&quot;);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an &quot;AS IS&quot; BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
MMOE CPU Run - Run inference on CPU multiple times
"""

import os
import sys
import numpy as np

# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.utils import (
    ensure_dir,
    save_results,
    parse_command_line_args,
    load_inputs_from_file,
    print_run_summary,
)

from config import (
    CPU_OUTPUT_FILE,
    CPU_ALL_OUTPUTS_FILE,
)


# Global variables for caching
_model = None
_params = None
_initialized = False


def get_model_and_params():
    """Get or create model and parameters (cached)."""
    global _model, _params
    
    if _model is None or _params is None:
        from utils import setup_model_and_params
        _model, _params = setup_model_and_params()
    
    return _model, _params


def run_cpu(num_runs=1, inputs=None):
    """Run CPU inference multiple times.
    
    Args:
        num_runs: Number of runs to execute
        inputs: List of input data for each run. If None, generates new inputs.
    """
    from utils import generate_mmoe_input, run_inference_and_save
    
    # Get cached model and params
    model, params = get_model_and_params()
    
    # Run inference multiple times with different inputs
    outputs = []
    
    for i in range(num_runs):
        if inputs is not None and i < len(inputs):
            x = inputs[i]
        else:
            x = generate_mmoe_input()
        
        print(f"[CPU] Running inference {i+1}/{num_runs}...", flush=True)
        output, execution_time = run_inference_and_save(model, params, x, "CPU", CPU_OUTPUT_FILE)
        outputs.append(output.copy())
        print(f"[CPU] Run {i+1}/{num_runs} completed in {execution_time:.2f} ms", flush=True)
    
    # Ensure output directory exists and save all outputs for comparison
    ensure_dir(CPU_ALL_OUTPUTS_FILE)
    all_outputs = np.stack(outputs, axis=0)
    all_inputs = np.stack([np.array(inp) for inp in (inputs if inputs else [generate_mmoe_input()] * num_runs)], axis=0) if inputs else None
    save_dict = {
        'outputs': all_outputs,
        'num_runs': num_runs
    }
    if all_inputs is not None:
        save_dict['inputs'] = all_inputs
    save_results(CPU_ALL_OUTPUTS_FILE, **save_dict)
    
    return outputs, inputs


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_command_line_args('CPU MMOE Run')
    
    num_runs = args.num_runs
    input_file = args.input_file
    
    # Check if input data is provided
    inputs = load_inputs_from_file(input_file, num_runs)

    # Run CPU inference
    outputs, inputs = run_cpu(num_runs, inputs)
    
    # Print results when called directly
    print_run_summary('CPU', num_runs, print_summary=False)
