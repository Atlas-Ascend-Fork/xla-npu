#!/usr/bin/env python3
# -*- coding:utf-8 -*-
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
        print("=== CPU and NPU Results Summary ===")
        print("=" * 60)

        if not os.path.exists(CPU_ALL_OUTPUTS_FILE):
            print(f"WARNING: CPU output file not found: {CPU_ALL_OUTPUTS_FILE}")
            return False

        if not os.path.exists(NPU_ALL_OUTPUTS_FILE):
            print(f"WARNING: NPU output file not found: {NPU_ALL_OUTPUTS_FILE}")
            return False

        # Load all results
        cpu_data = np.load(CPU_ALL_OUTPUTS_FILE)
        npu_data = np.load(NPU_ALL_OUTPUTS_FILE)

        all_match = True
        
        # Compare each operation
        for op_config in ops_configs:
            op_name = op_config['name'].replace(' ', '_').lower()
            
            print(f"\n{'=' * 60}")
            print(f"Comparing {op_config['name']} results")
            print(f"{'=' * 60}")
            
            if op_name not in cpu_data or op_name not in npu_data:
                print(f"WARNING: Results not found for {op_config['name']}")
                all_match = False
                continue
            
            cpu_outputs = cpu_data[op_name]['outputs']
            npu_outputs = npu_data[op_name]['outputs']
            
            # Compare each run
            for run_idx in range(min(len(cpu_outputs), len(npu_outputs), num_runs)):
                print(f"\n--- Run {run_idx + 1}/{num_runs} ---")
                
                cpu_output = np.array(cpu_outputs[run_idx])
                npu_output = np.array(npu_outputs[run_idx])
                
                # Use dummy input for comparison (same for both)
                dummy_input = np.zeros(1)
                match = compare_results(cpu_output, npu_output, dummy_input, dummy_input)
                all_match = all_match and match
        
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