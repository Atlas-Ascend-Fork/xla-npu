#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""
JAX Operations Test - Main entry point
Support running CPU and NPU in separate processes
"""

import os
import sys
import numpy as np

# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.base_runner import BaseRunner
from config import (
    DESCRIPTION, 
    DEFAULT_NUM_RUNS, 
    OPS_OUTPUT_DIR, 
    CPU_ALL_OUTPUTS_FILE, 
    NPU_ALL_OUTPUTS_FILE, 
    INPUT_DATA_FILE,
    ops_configs
)
from common.utils import (
    ensure_dir,
    compare_results,
    generate_and_save_inputs,
    get_output_files,
)


class OperationsRunner(BaseRunner):
    """Operations runner"""
    
    def compare_results(self, num_runs=1):
        """Compare all results from CPU and NPU runs.
        
        Args:
            num_runs: Number of runs to compare
        
        Returns:
            bool: True if all results match
        """
        print("\n" + "=" * 60)
        print("=== Execution Summary ===")
        print("=" * 60)

        if not os.path.exists(CPU_ALL_OUTPUTS_FILE):
            print(f"WARNING: CPU output file not found: {CPU_ALL_OUTPUTS_FILE}")
            return False

        if not os.path.exists(NPU_ALL_OUTPUTS_FILE):
            print(f"WARNING: NPU output file not found: {NPU_ALL_OUTPUTS_FILE}")
            return False

        # Load all results
        cpu_data = np.load(CPU_ALL_OUTPUTS_FILE, allow_pickle=True)
        npu_data = np.load(NPU_ALL_OUTPUTS_FILE, allow_pickle=True)

        # Calculate total execution time
        cpu_total_time = 0.0
        npu_total_time = 0.0
        
        for op_config in ops_configs:
            op_name = op_config['name'].replace(' ', '_').lower()
            
            if op_name in cpu_data:
                cpu_dict = cpu_data[op_name].item()
                cpu_avg_time = np.mean(cpu_dict['execution_times'])
                cpu_total_time += cpu_avg_time
            
            if op_name in npu_data:
                npu_dict = npu_data[op_name].item()
                npu_avg_time = np.mean(npu_dict['execution_times'])
                npu_total_time += npu_avg_time
        
        # Print total execution time
        print()
        print(f"[CPU] Total execution time: {cpu_total_time:.2f} ms")
        print(f"[NPU] Total execution time: {npu_total_time:.2f} ms")
        
        all_match = True
        match_results = []
        diff_details_list = []  # Store diff details for mismatches
        
        # Compare each operation
        for op_config in ops_configs:
            op_name = op_config['name'].replace(' ', '_').lower()
            
            if op_name not in cpu_data or op_name not in npu_data:
                match_results.append((op_config['name'], False, [], []))
                all_match = False
                continue
            
            # Get the actual dictionary from the numpy array
            cpu_dict = cpu_data[op_name].item()
            npu_dict = npu_data[op_name].item()
            
            cpu_outputs = cpu_dict['outputs']
            npu_outputs = npu_dict['outputs']
            
            # Compare each run
            op_match = True
            run_matches = []
            run_diff_details = []
            for run_idx in range(min(len(cpu_outputs), len(npu_outputs), num_runs)):
                cpu_output = np.array(cpu_outputs[run_idx])
                npu_output = np.array(npu_outputs[run_idx])
                
                # Use dummy input for comparison (same for both)
                dummy_input = np.zeros(1)
                match, diff_details = compare_results(cpu_output, npu_output, dummy_input, dummy_input)
                op_match = op_match and match
                run_matches.append(match)
                run_diff_details.append(diff_details)
            
            match_results.append((op_config['name'], op_match, run_matches))
            diff_details_list.append((op_config['name'], run_diff_details))
            all_match = all_match and op_match
        
        # Print summary of comparison results
        print("\n" + "=" * 60)
        print("=== Comparison Results ===")
        print("=" * 60)
        
        for i, (op_name, match, run_matches) in enumerate(match_results):
            status = "✅" if match else "❌"
            if num_runs > 1 and run_matches:
                run_status = ", ".join([f"Run {i+1}: {'✅' if m else '❌'}" for i, m in enumerate(run_matches)])
                print(f"{status} {op_name}: {run_status}")
            else:
                print(f"{status} {op_name}: {'Results match' if match else 'Results do not match'}")
            # Add blank line between operations (except after the last one)
            if i < len(match_results) - 1:
                print()
        
        # Print detailed diff information for mismatches
        if not all_match:
            print("\n" + "=" * 60)
            print("=== Difference Details ===")
            print("=" * 60)
            
            for op_name, run_diff_details in diff_details_list:
                has_mismatch = any(d is not None for d in run_diff_details)
                if has_mismatch:
                    print(f"\n[{op_name}]")
                    for run_idx, diff_details in enumerate(run_diff_details):
                        if diff_details is not None:
                            print(f"  ❌ Run {run_idx + 1} (MISMATCH):")
                            print(f"    Max absolute difference: {diff_details['max_absolute_diff']:.6e}")
                            print(f"    Max relative difference: {diff_details['max_relative_diff']:.6e}")
                            print(f"    Mean absolute difference: {diff_details['mean_absolute_diff']:.6e}")
                            print(f"    Max diff location: {diff_details['max_diff_location']}")
                            print(f"    CPU value at max diff: {diff_details['max_diff_cpu_value']:.6e}")
                            print(f"    NPU value at max diff: {diff_details['max_diff_npu_value']:.6e}")

                            # Print actual values if available (small arrays)
                            if 'cpu_values' in diff_details:
                                print(f"    CPU values: {diff_details['cpu_values']}")
                                print(f"    NPU values: {diff_details['npu_values']}")
                                print(f"    Diff values: {diff_details['diff_values']}")
        
        return all_match
    
    def generate_inputs(self, num_runs=1):
        """Generate input data for each run.
        
        Args:
            num_runs: Number of runs to generate inputs for
        
        Returns:
            tuple: (inputs, input_file)
        """
        # Generate input data for each run (same for CPU/NPU)
        print("Generating input data...")
        inputs_dict = generate_and_save_inputs(INPUT_DATA_FILE, num_runs, ops_configs)
        
        return inputs_dict, INPUT_DATA_FILE
    
    def print_config(self, args):
        """Print operations configuration.
        
        Args:
            args: Parsed arguments
        """
        print(DESCRIPTION)
        print("=" * 50)
        print(f"Configuration:")
        print(f"  Number of runs: {args.num_runs}")
        print(f"  Skip CPU: {args.skip_cpu}")
        print(f"  Skip NPU: {args.skip_npu}")


if __name__ == "__main__":
    # Create config object with run number support
    class Config:
        DESCRIPTION = DESCRIPTION
        DEFAULT_NUM_RUNS = DEFAULT_NUM_RUNS
        
        def __init__(self, run_number=1):
            # Update output paths with run number
            output_files = get_output_files(OPS_OUTPUT_DIR, run_number=run_number)
            self.CPU_ALL_OUTPUTS_FILE = output_files['cpu_all_outputs_file']
            self.NPU_ALL_OUTPUTS_FILE = output_files['npu_all_outputs_file']
            self.INPUT_DATA_FILE = output_files['input_data_file']
    
    # Create config with run number 1 (can be extended to support multiple runs)
    config = Config(run_number=1)
    
    # Create and run runner
    runner = OperationsRunner(config)
    runner.main()