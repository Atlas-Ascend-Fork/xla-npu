# -*- coding:utf-8 -*-
"""
MMOE Test - Main entry point for CPU and NPU comparison test
Support one-time compilation and multiple runs in each process
"""

import os
import sys
import subprocess
import numpy as np
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_inference(run_type, num_runs=1, test_idx=1):
    """Run inference on CPU or NPU for multiple runs in a single process.
    
    Args:
        run_type: 'CPU' or 'NPU'
        num_runs: Number of runs to execute
        test_idx: Test index (1-based), used to generate different inputs
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_name = f"{run_type.lower()}_test.py"
    script_path = os.path.join(script_dir, script_name)
    
    # Set environment
    env = os.environ.copy()
    env['NUM_RUNS'] = str(num_runs)
    env['TEST_IDX'] = str(test_idx)
    
    print(f"\n{'=' * 60}")
    print(f"Running {run_type} inference (test_idx={test_idx}, {num_runs} runs in single process)...")
    print(f"{'=' * 60}")
    
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=False,
        env=env
    )
    
    return result.returncode == 0


def compare_all_results(num_runs=1):
    """Compare all results from CPU and NPU runs.
    
    Args:
        num_runs: Number of runs to compare
    """
    from model import compare_results, print_results, CONFIG_BATCH_SIZE, CONFIG_NUM_TASKS
    
    cpu_output_path = "/tmp/mmoe_cpu_all_outputs.npz"
    npu_output_path = "/tmp/mmoe_npu_all_outputs.npz"
    
    if not os.path.exists(cpu_output_path):
        print(f"WARNING: CPU output file not found: {cpu_output_path}")
        return False
    
    if not os.path.exists(npu_output_path):
        print(f"WARNING: NPU output file not found: {npu_output_path}")
        return False
    
    print("\n" + "=" * 60)
    print("=== CPU and NPU Results Summary ===")
    print("=" * 60)
    
    # Load all results
    cpu_data = np.load(cpu_output_path)
    npu_data = np.load(npu_output_path)
    
    cpu_outputs = cpu_data['outputs']
    npu_outputs = npu_data['outputs']
    cpu_runs = int(cpu_data['num_runs'])
    npu_runs = int(npu_data['num_runs'])
    
    print(f"\nInput shape: {cpu_data['input'].shape}")
    print(f"Output shape per run: {cpu_outputs[0].shape}")
    print(f"Expected output shape: ({CONFIG_BATCH_SIZE}, {CONFIG_NUM_TASKS})")
    print(f"CPU runs: {cpu_runs}, NPU runs: {npu_runs}")
    
    # Compare each run
    all_match = True
    for run_idx in range(min(cpu_runs, npu_runs, num_runs)):
        print(f"\n{'=' * 60}")
        print(f"--- Run {run_idx+1}/{num_runs} ---")
        print(f"{'=' * 60}")
        
        cpu_output = cpu_outputs[run_idx]
        npu_output = npu_outputs[run_idx]
        
        print_results(cpu_output, "CPU", CONFIG_BATCH_SIZE, CONFIG_NUM_TASKS)
        print_results(npu_output, "NPU", CONFIG_BATCH_SIZE, CONFIG_NUM_TASKS)
        
        match = compare_results(cpu_output, npu_output, 
                               cpu_data['input'], cpu_data['input'])
        all_match = all_match and match
    
    # Cleanup
    print("\n" + "=" * 60)
    print("Cleaning up temporary files...")
    print("=" * 60)
    
    files_to_remove = [
        "/tmp/mmoe_cpu_all_outputs.npz",
        "/tmp/mmoe_npu_all_outputs.npz"
    ]
    
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"  Removed: {file_path}")
    
    return all_match


def main():
    """Run CPU and NPU tests and compare results."""
    parser = argparse.ArgumentParser(description='MMOE JAX Implementation Test')
    parser.add_argument('--num-tests', type=int, default=1,
                       help='Number of test iterations (default: 1)')
    parser.add_argument('--skip-cpu', action='store_true',
                       help='Skip CPU inference, only run NPU')
    parser.add_argument('--skip-npu', action='store_true',
                       help='Skip NPU inference, only run CPU')
    parser.add_argument('--init', action='store_true', default=False,
                       help='Initialize model parameters (default: False, use existing params)')
    
    args = parser.parse_args()
    
    print("MMOE JAX Implementation Test")
    print("=" * 50)
    print(f"Configuration:")
    print(f"  Number of tests: {args.num_tests}")
    print(f"  Skip CPU: {args.skip_cpu}")
    print(f"  Skip NPU: {args.skip_npu}")
    print(f"  Init Params: {args.init}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    params_file = os.path.join(script_dir, "model_params.npz")
    
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
                sys.exit(1)
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
            sys.exit(1)
        else:
            print("\n" + "=" * 60)
            print(f"Using existing parameter file: {params_file}")
            print("=" * 60)
    
    # Run multiple test iterations
    all_passed = True
    for test_idx in range(1, args.num_tests + 1):
        print("\n" + "=" * 60)
        print(f"TEST ITERATION {test_idx}/{args.num_tests}")
        print("=" * 60)
        
        # Run CPU inference (single run in process)
        if not args.skip_cpu:
            if not run_inference("CPU", num_runs=1, test_idx=test_idx):
                print(f"ERROR: CPU inference failed in test {test_idx}")
                sys.exit(1)
        
        # Run NPU inference (single run in process)
        if not args.skip_npu:
            if not run_inference("NPU", num_runs=1, test_idx=test_idx):
                print(f"ERROR: NPU inference failed in test {test_idx}")
                sys.exit(1)
        
        # Comparison phase (only if both CPU and NPU ran)
        if not args.skip_cpu and not args.skip_npu:
            test_passed = compare_all_results(num_runs=1)
            all_passed = all_passed and test_passed
            
            print(f"\n{'=' * 60}")
            if test_passed:
                print(f"✅ TEST {test_idx}/{args.num_tests} PASSED")
            else:
                print(f"❌ TEST {test_idx}/{args.num_tests} FAILED")
            print("=" * 60)
        else:
            print(f"\n{'=' * 60}")
            print(f"⚠️  Skipped comparison for test {test_idx} (CPU or NPU was skipped)")
            print("=" * 60)
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    if all_passed:
        print(f"✅ ALL {args.num_tests} TESTS PASSED")
    else:
        print(f"❌ SOME TESTS FAILED")
    print("=" * 60)


if __name__ == "__main__":
    main()
