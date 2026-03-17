# -*- coding:utf-8 -*-
"""
MMOE Test - Main entry point for CPU and NPU comparison test
Support one-time compilation and multiple runs in each process
"""

import os
import subprocess
import sys
import numpy as np

# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.base_runner import BaseRunner
from config import (
    DESCRIPTION, 
    DEFAULT_NUM_RUNS, 
    MMOE_OUTPUT_DIR, 
    TEST_OUTPUT_DIR, 
    CPU_OUTPUT_FILE, 
    NPU_OUTPUT_FILE, 
    CPU_ALL_OUTPUTS_FILE, 
    NPU_ALL_OUTPUTS_FILE, 
    INPUT_DATA_FILE, 
    CONFIG_BATCH_SIZE, 
    CONFIG_NUM_TASKS,
    CONFIG_MODEL_PARAMS_FILE
)
from common.utils import (
    compare_results,
    print_results,
    print_input_stats,
    generate_and_save_inputs,
)
from utils import generate_mmoe_input, ensure_dir


class MMOERunner(BaseRunner):
    """MMOE runner"""
    
    def compare_results(self, num_runs=1):
        """Compare all results from CPU and NPU runs.
        
        Args:
            num_runs: Number of runs to compare
        
        Returns:
            bool: True if all results match
        """
        cpu_output_path = CPU_ALL_OUTPUTS_FILE
        npu_output_path = NPU_ALL_OUTPUTS_FILE

        if not os.path.exists(cpu_output_path):
            print(f"WARNING: CPU output file not found: {cpu_output_path}")
            return False

        if not os.path.exists(npu_output_path):
            print(f"WARNING: NPU output file not found: {npu_output_path}")
            return False

        print("\n" + "=" * 60)
        print("=== Execution Summary ===")
        print("=" * 60)

        # Load all results
        cpu_data = np.load(cpu_output_path)
        npu_data = np.load(npu_output_path)

        cpu_outputs = cpu_data['outputs']
        npu_outputs = npu_data['outputs']
        cpu_runs = int(cpu_data['num_runs'])
        npu_runs = int(npu_data['num_runs'])
        
        # Load execution times if available
        cpu_total_time = 0.0
        npu_total_time = 0.0
        
        # Check if execution times are available in individual output files
        for run_idx in range(min(cpu_runs, npu_runs, num_runs)):
            try:
                cpu_run_data = np.load(CPU_OUTPUT_FILE)
                if 'execution_time' in cpu_run_data:
                    cpu_total_time += float(cpu_run_data['execution_time'])
            except:
                pass
            
            try:
                npu_run_data = np.load(NPU_OUTPUT_FILE)
                if 'execution_time' in npu_run_data:
                    npu_total_time += float(npu_run_data['execution_time'])
            except:
                pass
        
        # Load inputs for each run
        cpu_inputs = cpu_data.get('inputs', None)
        npu_inputs = npu_data.get('inputs', None)
        
        print(f"\nOutput shape per run: {cpu_outputs[0].shape}")
        print(f"Expected output shape: ({CONFIG_BATCH_SIZE}, {CONFIG_NUM_TASKS})")
        print(f"CPU runs: {cpu_runs}, NPU runs: {npu_runs}")
        
        # Print total execution time
        print()
        if cpu_total_time > 0:
            print(f"[CPU] Total execution time: {cpu_total_time:.2f} ms")
        if npu_total_time > 0:
            print(f"[NPU] Total execution time: {npu_total_time:.2f} ms")

        # Compare each run
        all_match = True
        match_results = []
        for run_idx in range(min(cpu_runs, npu_runs, num_runs)):
            cpu_output = cpu_outputs[run_idx]
            npu_output = npu_outputs[run_idx]
            
            # Print output info for this run
            print(f"\n{'=' * 60}")
            print(f"--- Run {run_idx + 1}/{num_runs} ---")
            print(f"{'=' * 60}")
            
            # Print CPU output stats
            print(f"[CPU] Output:")
            print(f"[CPU]   Shape: {cpu_output.shape}")
            print(f"[CPU]   Min: {cpu_output.min():.6f}")
            print(f"[CPU]   Max: {cpu_output.max():.6f}")
            print(f"[CPU]   Mean: {cpu_output.mean():.6f}")
            
            # Print NPU output stats
            print(f"\n[NPU] Output:")
            print(f"[NPU]   Shape: {npu_output.shape}")
            print(f"[NPU]   Min: {npu_output.min():.6f}")
            print(f"[NPU]   Max: {npu_output.max():.6f}")
            print(f"[NPU]   Mean: {npu_output.mean():.6f}")
            
            # Use corresponding inputs for comparison
            cpu_input = cpu_inputs[run_idx] if cpu_inputs is not None else None
            npu_input = npu_inputs[run_idx] if npu_inputs is not None else None
            
            # Calculate differences
            diff = np.abs(cpu_output - npu_output)
            rel_diff = diff / (np.abs(cpu_output) + 1e-8)
            
            print(f"\nDifference statistics:")
            print(f"  Max absolute difference: {diff.max():.6e}")
            print(f"  Mean absolute difference: {diff.mean():.6e}")
            print(f"  Max relative difference: {rel_diff.max():.6e}")
            print(f"  Mean relative difference: {rel_diff.mean():.6e}")
            
            if cpu_input is not None and npu_input is not None:
                match, diff_details = compare_results(cpu_output, npu_output, cpu_input, npu_input)
            else:
                match, diff_details = compare_results(cpu_output, npu_output,
                                       cpu_data.get('input', np.zeros(1)),
                                       npu_data.get('input', np.zeros(1)))

            print(f"\nResults match: {match}")
            match_results.append((run_idx + 1, match))
            all_match = all_match and match
        
        # Print comparison results summary
        if match_results:
            print("\n" + "=" * 60)
            print("=== Comparison Results Summary ===")
            print("=" * 60)
            for run_num, match in match_results:
                status = "✅" if match else "❌"
                print(f"{status} Run {run_num}/{len(match_results)}: {'Results match' if match else 'Results do not match'}")
        
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
        inputs = generate_and_save_inputs(INPUT_DATA_FILE, num_runs, generate_mmoe_input=generate_mmoe_input)
        
        return inputs, INPUT_DATA_FILE
    
    def add_args(self, parser):
        """Add MMOE-specific arguments to parser.
        
        Args:
            parser: ArgumentParser object
        """
        parser.add_argument('--init', action='store_true', default=False,
                           help='Initialize model parameters (default: False, use existing params)')
    
    def print_config(self, args):
        """Print MMOE configuration.
        
        Args:
            args: Parsed arguments
        """
        print(DESCRIPTION)
        print("=" * 50)
        print(f"Configuration:")
        print(f"  Number of runs: {args.num_runs}")
        print(f"  Skip CPU: {args.skip_cpu}")
        print(f"  Skip NPU: {args.skip_npu}")
        print(f"  Init Params: {args.init}")
    
    def initialize(self, args):
        """MMOE-specific initialization.
        
        Args:
            args: Parsed arguments
        
        Returns:
            bool: True if initialization succeeded
        """
        script_dir = self.script_dir
        params_file = os.path.join(script_dir, CONFIG_MODEL_PARAMS_FILE)

        # Initialize model parameters only if --init flag is provided
        if args.init:
            print("\n" + "=" * 60)
            print("Initializing model parameters...")
            print("=" * 60)

            init_script = os.path.join(script_dir, "init_params.py")
            if os.path.exists(init_script):
                result = subprocess.run(
                    [sys.executable, init_script],
                    capture_output=False,
                    cwd=script_dir
                )
                if result.returncode != 0:
                    print("ERROR: Model parameter initialization failed")
                    return False
            else:
                print(f"WARNING: init script not found: {init_script}")
                print("Continuing without parameter initialization...")
        else:
            # Check if params file exists
            if not os.path.exists(params_file):
                print("\n" + "=" * 60)
                print(f"WARNING: Parameter file not found: {params_file}")
                print("Please run with --init flag to generate parameters first")
                print("=" * 60)
                return False
            else:
                print("\n" + "=" * 60)
                print(f"Using existing parameter file: {params_file}")
                print("=" * 60)
        
        return True


if __name__ == "__main__":
    # Create config object
    class Config:
        DESCRIPTION = DESCRIPTION
        DEFAULT_NUM_RUNS = DEFAULT_NUM_RUNS
        CPU_ALL_OUTPUTS_FILE = CPU_ALL_OUTPUTS_FILE
        NPU_ALL_OUTPUTS_FILE = NPU_ALL_OUTPUTS_FILE
        INPUT_DATA_FILE = INPUT_DATA_FILE
    
    config = Config()
    
    # Create and run runner
    runner = MMOERunner(config)
    runner.main()