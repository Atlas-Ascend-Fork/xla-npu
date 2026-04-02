# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Operations CPU Run - Run operations on CPU multiple times
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
    OPS_OUTPUT_DIR,
    CPU_ALL_OUTPUTS_FILE,
    ops_configs,
)


def run_cpu(num_runs=1, inputs_dict=None, run_number=1):
    """Run CPU operations multiple times.
    
    Args:
        num_runs: Number of runs to execute
        inputs_dict: Dictionary of input data for each operation. If None, generates new inputs.
        run_number: Run number for output directory structure
    """
    from utils import run_operation
    
    # Run each operation
    all_results = {}
    
    for op_config in ops_configs:
        # Get inputs for this operation
        op_inputs_list = None
        if inputs_dict:
            op_name = op_config['name'].replace(' ', '_').lower()
            if op_name in inputs_dict:
                op_inputs_list = inputs_dict[op_name]
        
        # Run operation on CPU
        outputs, _, execution_times = run_operation(op_config, num_runs, 'CPU', op_inputs_list, run_number)
        
        # Save results for this operation
        op_name = op_config['name'].replace(' ', '_').lower()
        all_results[op_name] = {
            'outputs': np.array(outputs),
            'execution_times': np.array(execution_times)
        }
    
    # Ensure output directory exists and save all outputs for comparison
    ensure_dir(CPU_ALL_OUTPUTS_FILE)
    save_results(CPU_ALL_OUTPUTS_FILE, **all_results)
    
    return all_results


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_command_line_args('CPU Operations Run')
    
    num_runs = args.num_runs
    input_file = args.input_file
    run_number = args.run_number
    
    # Check if input data is provided
    inputs_dict = load_inputs_from_file(input_file, num_runs, ops_configs)

    # Run CPU operations
    all_results = run_cpu(num_runs, inputs_dict, run_number)
    
    # Print results when called directly
    print_run_summary('CPU', num_runs, all_results, print_summary=False)