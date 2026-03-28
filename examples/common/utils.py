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

import os
import time
import sys
import numpy as np
import jax
import jax.numpy as jnp

from common.config import (
    NPU_PLATFORM_NAME,
    NPU_LIBRARY_NAME,
    ENV_PJRT_NAMES_AND_LIBRARY_PATHS,
    ENV_JAX_PLATFORMS,
    COMPARISON_TOLERANCE,
    MAX_VALUES_TO_PRINT,
    PRINT_SEPARATOR,
    OUTPUT_SAMPLE_SIZE,
    OUTPUT_DIR,
)


def setup_npu_environment():
    """Setup NPU environment variables for JAX."""
    library_path = os.path.abspath(NPU_LIBRARY_NAME)
    os.environ[ENV_PJRT_NAMES_AND_LIBRARY_PATHS] = f"{NPU_PLATFORM_NAME}:{library_path}"
    os.environ[ENV_JAX_PLATFORMS] = NPU_PLATFORM_NAME


def setup_device_environment(device):
    """Setup environment variables for the specified device.
    
    Args:
        device: Device to setup ('CPU' or 'NPU')
    """
    if device == 'NPU':
        # Ensure NPU is initialized
        setup_npu_environment()
    else:
        # Ensure only CPU is used for CPU runs
        os.environ[ENV_JAX_PLATFORMS] = 'cpu'
        # Clear any NPU plugin paths
        if ENV_PJRT_NAMES_AND_LIBRARY_PATHS in os.environ:
            del os.environ[ENV_PJRT_NAMES_AND_LIBRARY_PATHS]


def ensure_dir(file_path):
    """Ensure the directory for the given file path exists.
    
    Args:
        file_path: Path to a file
    """
    output_dir = os.path.dirname(file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)


def generate_input(shape, seed=None):
    """Generate random input data.
    
    Args:
        shape: Shape of the input tensor
        seed: Random seed for reproducibility. If None, uses random seed.
    
    Returns:
        np.ndarray: Random input tensor
    """
    if seed is None:
        seed = int(np.random.randint(0, 10000))
    
    rng = np.random.RandomState(seed)
    x = rng.randn(*shape).astype(np.float32)
    return x


def time_function(func, *args, **kwargs):
    """Run a function and measure execution time.
    
    Args:
        func: Function to run
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
    
    Returns:
        tuple: (result, execution_time_ms)
    """
    # Warmup run
    func(*args, **kwargs)
    
    # Measure actual execution time
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    
    execution_time_ms = (end_time - start_time) * 1000
    return result, execution_time_ms


def save_results(file_path, **kwargs):
    """Save results to npz file.
    
    Args:
        file_path: Path to save results
        **kwargs: Key-value pairs to save
    """
    ensure_dir(file_path)
    np.savez(file_path, **kwargs)


def compare_results(output_cpu, output_npu, x_cpu, x_npu):
    """Compare CPU and NPU results.
    
    Args:
        output_cpu: CPU output
        output_npu: NPU output
        x_cpu: CPU input
        x_npu: NPU input
        
    Returns:
        tuple: (is_match, diff_details) where diff_details is a dict with difference info or None
    """
    output_cpu = np.array(output_cpu)
    output_npu = np.array(output_npu)
    
    is_match = np.allclose(output_cpu, output_npu, rtol=COMPARISON_TOLERANCE, atol=COMPARISON_TOLERANCE)
    
    diff_details = None
    if not is_match:
        diff = np.abs(output_cpu - output_npu)
        rel_diff = diff / (np.abs(output_cpu) + 1e-10)
        
        max_diff_idx = np.unravel_index(np.argmax(diff), diff.shape)
        max_rel_diff_idx = np.unravel_index(np.argmax(rel_diff), rel_diff.shape)
        
        diff_details = {
            'max_absolute_diff': float(np.max(diff)),
            'max_relative_diff': float(np.max(rel_diff)),
            'mean_absolute_diff': float(np.mean(diff)),
            'max_diff_location': max_diff_idx,
            'max_diff_cpu_value': float(output_cpu[max_diff_idx]),
            'max_diff_npu_value': float(output_npu[max_diff_idx]),
            'cpu_output_shape': output_cpu.shape,
            'npu_output_shape': output_npu.shape,
        }
        
        # Print actual values if total elements <= MAX_VALUES_TO_PRINT
        total_elements = output_cpu.size
        if total_elements <= MAX_VALUES_TO_PRINT:
            diff_details['cpu_values'] = output_cpu.flatten().tolist()
            diff_details['npu_values'] = output_npu.flatten().tolist()
            diff_details['diff_values'] = diff.flatten().tolist()
    
    return is_match, diff_details


def print_input_stats(x, run_idx=None, device_name=None):
    """Print input data statistics.
    
    Args:
        x: Input tensor
        run_idx: Run index (optional, not used in output)
        device_name: Device name (optional, not used in output)
    """
    x_arr = np.array(x)
    
    print(f"\nInput:")
    print(f"  Shape: {x_arr.shape}")
    print(f"  Min: {x_arr.min():.6f}")
    print(f"  Max: {x_arr.max():.6f}")
    print(f"  Mean: {x_arr.mean():.6f}")
    print(f"  Std: {x_arr.std():.6f}")


def print_results(output, device_name, batch_size=None, num_tasks=None):
    """Print inference results.
    
    Args:
        output: Output tensor
        device_name: Device name ('CPU' or 'NPU')
        batch_size: Batch size (optional)
        num_tasks: Number of tasks (optional)
    """
    print(f"\n[{device_name}] Output:")
    print(f"[{device_name}]   Shape: {output.shape}")
    print(f"[{device_name}]   Min: {output.min():.6f}")
    print(f"[{device_name}]   Max: {output.max():.6f}")
    print(f"[{device_name}]   Mean: {output.mean():.6f}")
    print(f"[{device_name}]   Std: {output.std():.6f}")


def print_run_summary(device_name, num_runs, results=None, print_summary=True):
    """Print run summary.
    
    Args:
        device_name: Device name ('CPU' or 'NPU')
        num_runs: Number of runs
        results: Dictionary of results (optional, for ops module)
        print_summary: Whether to print the summary
    
    Returns:
        float: Total execution time (ms) if results are provided
    """
    total_time = None
    print(f"\n[{device_name}] All operations completed and saved")
    
    if results:
        total_time = 0.0
        for op_name, result in results.items():
            avg_time = np.mean(result['execution_times'])
            total_time += avg_time
        if print_summary:
            print(f"\n[{device_name}] Results summary:")
            print(f"  Total execution time: {total_time:.2f} ms")
            print()
    elif print_summary:
        print(f"\n[{device_name}] Results summary:")
        print(f"  Total runs: {num_runs}")
        print()
    
    return total_time


# ============================================================================
# Configuration Utilities
# ============================================================================


def setup_path():
    """Setup Python path to include parent directory."""
    # Add parent directory to Python path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_output_dirs(base_output_dir, subdir_name):
    """Create output directories for a module.
    
    Args:
        base_output_dir: Base output directory
        subdir_name: Subdirectory name for the module
    
    Returns:
        tuple: (module_output_dir, test_output_dir)
    """
    module_output_dir = f"{base_output_dir}/{subdir_name}"
    test_output_dir = f"{module_output_dir}/test"
    return module_output_dir, test_output_dir


def get_output_files(module_output_dir, test_output_dir=None, run_number=None):
    """Get output file paths for a module.
    
    Args:
        module_output_dir: Module output directory
        test_output_dir: Test output directory (optional)
        run_number: Run number (optional)
    
    Returns:
        dict: Output file paths
    """
    # Use test_output_dir if provided, otherwise use module_output_dir
    output_dir = test_output_dir if test_output_dir else module_output_dir
    
    # Add run number to directory if provided
    if run_number is not None:
        output_dir = os.path.join(output_dir, str(run_number))
    
    return {
        'cpu_output_file': f"{output_dir}/cpu_output.npz",
        'npu_output_file': f"{output_dir}/npu_output.npz",
        'cpu_all_outputs_file': f"{output_dir}/cpu_all_outputs.npz",
        'npu_all_outputs_file': f"{output_dir}/npu_all_outputs.npz",
        'input_data_file': f"{output_dir}/input_data.npz"
    }


# ============================================================================
# Run Utilities
# ============================================================================


def parse_command_line_args(description):
    """Parse command line arguments for run scripts.
    
    Args:
        description: Script description
    
    Returns:
        parsed arguments
    """
    import argparse
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--num-runs', type=int, default=1,
                        help='Number of runs to execute')
    parser.add_argument('--input-file', type=str, default=None,
                        help='Path to input data file')
    parser.add_argument('--run-number', type=int, default=1,
                        help='Run number for output directory structure')
    return parser.parse_args()


def load_inputs_from_file(input_file, num_runs, ops_configs=None):
    """Load inputs from file.
    
    Args:
        input_file: Path to input data file
        num_runs: Number of runs
        ops_configs: Operation configurations (for ops module)
    
    Returns:
        inputs_dict or inputs list
    """
    if not input_file or not os.path.exists(input_file):
        return None
    
    data = np.load(input_file)
    
    # For MMOE
    if 'inputs' in data:
        inputs = list(data['inputs'])
        num_runs_in_file = int(data.get('num_runs', len(inputs)))
        return inputs[:min(num_runs, num_runs_in_file)]
    elif 'input' in data:
        return [data['input']] * num_runs
    # For Ops
    elif ops_configs:
        inputs_dict = {}
        for op_config in ops_configs:
            op_name = op_config['name'].replace(' ', '_').lower()
            inputs_dict[op_name] = []
            
            for i in range(num_runs):
                op_inputs = {}
                for key in data.files:
                    if f"{op_name}_input_{i}_" in key:
                        op_inputs[key.replace(f"{op_name}_input_{i}_", '')] = data[key]
                if op_inputs:
                    inputs_dict[op_name].append(op_inputs)
        return inputs_dict
    
    return None


def generate_and_save_inputs(input_file, num_runs, ops_configs=None, generate_mmoe_input=None):
    """Generate and save input data.
    
    Args:
        input_file: Path to save input data
        num_runs: Number of runs
        ops_configs: Operation configurations (for ops module)
        generate_mmoe_input: Function to generate MMOE input
    
    Returns:
        inputs_dict or inputs list
    """
    ensure_dir(input_file)
    
    # For Ops
    if ops_configs:
        inputs_dict = {}
        save_dict = {'num_runs': num_runs}
        
        for op_config in ops_configs:
            op_name = op_config['name'].replace(' ', '_').lower()
            inputs_dict[op_name] = []
            
            for i in range(num_runs):
                op_inputs = op_config['inputs']()
                inputs_dict[op_name].append(op_inputs)
                
                # Save inputs for this operation and run
                for key, value in op_inputs.items():
                    save_dict[f"{op_name}_input_{i}_{key}"] = value
        
        np.savez(input_file, **save_dict)
        return inputs_dict
    # For MMOE
    elif generate_mmoe_input:
        inputs = []
        for i in range(num_runs):
            inputs.append(generate_mmoe_input())
        
        # Save inputs
        save_dict = {
            'inputs': np.stack(inputs, axis=0),
            'num_runs': num_runs
        }
        np.savez(input_file, **save_dict)
        return inputs
    
    return None
