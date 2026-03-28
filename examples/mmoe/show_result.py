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
MMOE Full Output Comparison - Show and align all CPU and NPU outputs
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    CONFIG_BATCH_SIZE,
    CONFIG_NUM_TASKS,
    CPU_ALL_OUTPUTS_FILE,
    NPU_ALL_OUTPUTS_FILE,
    INPUT_DATA_FILE,
    COMPARISON_TOLERANCE,
    PRINT_SEPARATOR,
)


def format_array(arr, name="", width=12, precision=6):
    """Format array for aligned display.
    
    Args:
        arr: Array to format
        name: Array name
        width: Field width
        precision: Decimal precision
    
    Returns:
        str: Formatted string
    """
    flat = arr.flatten()
    lines = []
    if name:
        lines.append(f"{name}:")
    
    # Print in rows of 4 elements
    for i in range(0, len(flat), 4):
        row = flat[i:i+4]
        formatted = "  ".join(f"{x:>{width}.{precision}f}" for x in row)
        lines.append(f"  [{i:>3}]: {formatted}")
    
    return "\n".join(lines)


def print_output_stats(output, device_name):
    """Print output statistics.
    
    Args:
        output: Output array
        device_name: Device name ('CPU' or 'NPU')
    """
    print("\n" + PRINT_SEPARATOR)
    print(f"{device_name} Output Statistics")
    print(PRINT_SEPARATOR)
    print(f"  Shape: {output.shape}")
    print(f"  Min: {output.min():.6f}")
    print(f"  Max: {output.max():.6f}")
    print(f"  Mean: {output.mean():.6f}")
    print(f"  Std: {output.std():.6f}")


def show_single_output(data_file, device_name, run_idx=0, show_all_elements=False, max_elements=100):
    """Show single device output (CPU or NPU only).
    
    Args:
        data_file: Path to output data file
        device_name: Device name ('CPU' or 'NPU')
        run_idx: Run index to show
        show_all_elements: Whether to show all elements
        max_elements: Maximum elements to show if not showing all
    
    Returns:
        bool: True if successful
    """
    if not os.path.exists(data_file):
        print(f"\n⚠️  WARNING: {device_name} output file not found: {data_file}")
        print(f"    Skipping {device_name} output display.")
        print(f"    Run 'python main.py' first to generate {device_name} outputs.")
        return False
    
    # Load data
    data = np.load(data_file)
    outputs = data['outputs']
    inputs = data.get('inputs', None)
    
    num_runs = len(outputs)
    
    if run_idx >= num_runs:
        print(f"ERROR: Run index {run_idx} out of range (total runs: {num_runs})")
        return False
    
    print(f"\n{'=' * 80}")
    print(f"{device_name} Output (Run {run_idx + 1}/{num_runs})")
    print(f"{'=' * 80}")
    print(f"Output shape: {outputs[run_idx].shape}")
    
    # Get specific run data
    output = outputs[run_idx]
    input_data = inputs[run_idx] if inputs is not None else None
    
    # Print input info
    if input_data is not None:
        print("\n" + PRINT_SEPARATOR)
        print("Input Data Statistics")
        print(PRINT_SEPARATOR)
        print(f"  Shape: {input_data.shape}")
        print(f"  Min: {input_data.min():.6f}")
        print(f"  Max: {input_data.max():.6f}")
        print(f"  Mean: {input_data.mean():.6f}")
        print(f"  Std: {input_data.std():.6f}")
    
    # Print output statistics
    print_output_stats(output, device_name)
    
    # Show full output (flattened)
    print("\n" + "=" * 80)
    print(f"{device_name} Full Output")
    print("=" * 80)
    print(format_array(output, device_name, width=12, precision=6))
    
    # Show first N elements in detail
    elements_to_show = output.size if show_all_elements else min(max_elements, output.size)
    flat_output = output.flatten()
    
    print("\n" + "=" * 80)
    print(f"{device_name} Output Detail (first {elements_to_show} elements)")
    print("=" * 80)
    print(f"{'Index':>8} | {'Value':>16}")
    print("-" * 30)
    
    for i in range(elements_to_show):
        print(f"{i:>8} | {flat_output[i]:>16.6e}")
    
    if not show_all_elements and elements_to_show < output.size:
        print(f"... ({output.size - elements_to_show} more elements)")
    
    # Summary by last dimension (if multi-dimensional)
    print("\n" + "=" * 80)
    print(f"{device_name} Per-Channel Summary (last dimension)")
    print("=" * 80)
    
    if output.ndim >= 2:
        # Treat last dimension as channels
        num_channels = output.shape[-1]
        for channel_idx in range(num_channels):
            # Extract all elements for this channel
            channel_data = output[..., channel_idx]
            print(f"\nChannel {channel_idx} (shape {channel_data.shape}):")
            print(f"  Elements: {channel_data.size}")
            print(f"  Min: {channel_data.min():.6f}")
            print(f"  Max: {channel_data.max():.6f}")
            print(f"  Mean: {channel_data.mean():.6f}")
            print(f"  Std: {channel_data.std():.6f}")
    else:
        # 1D output
        print(f"\nOutput (1D, {output.size} elements):")
        print(f"  Min: {output.min():.6f}")
        print(f"  Max: {output.max():.6f}")
        print(f"  Mean: {output.mean():.6f}")
        print(f"  Std: {output.std():.6f}")
    
    return True


def show_full_comparison(run_idx=0, show_all_elements=False, max_elements=100, 
                         show_cpu=True, show_npu=True):
    """Show full comparison between CPU and NPU outputs.
    
    Args:
        run_idx: Run index to compare
        show_all_elements: Whether to show all elements
        max_elements: Maximum elements to show if not showing all
        show_cpu: Whether to show CPU output
        show_npu: Whether to show NPU output
    
    Returns:
        bool: True if comparison was performed
    """
    # Check which files exist
    cpu_exists = os.path.exists(CPU_ALL_OUTPUTS_FILE)
    npu_exists = os.path.exists(NPU_ALL_OUTPUTS_FILE)
    
    # If only showing one device, use single output display
    if show_cpu and not show_npu:
        return show_single_output(CPU_ALL_OUTPUTS_FILE, "CPU", run_idx, show_all_elements, max_elements)
    
    if show_npu and not show_cpu:
        return show_single_output(NPU_ALL_OUTPUTS_FILE, "NPU", run_idx, show_all_elements, max_elements)
    
    # Both CPU and NPU should be shown - do comparison
    print("\n" + "=" * 80)
    print("MMOE CPU vs NPU Full Output Comparison")
    print("=" * 80)
    
    # Check if files exist
    if not cpu_exists:
        print(f"\n⚠️  WARNING: CPU output file not found: {CPU_ALL_OUTPUTS_FILE}")
        print("    Skipping CPU output and comparison.")
        print("    Run 'python main.py' first to generate CPU outputs.")
    
    if not npu_exists:
        print(f"\n⚠️  WARNING: NPU output file not found: {NPU_ALL_OUTPUTS_FILE}")
        print("    Skipping NPU output and comparison.")
        print("    Run 'python main.py' first to generate NPU outputs.")
    
    # If neither exists, exit
    if not cpu_exists and not npu_exists:
        print("\n❌ ERROR: No output files found. Please run 'python main.py' first.")
        return False
    
    # If only one exists, show that one
    if cpu_exists and not npu_exists:
        print("\nℹ️  Only CPU output available. Showing CPU output only.")
        return show_single_output(CPU_ALL_OUTPUTS_FILE, "CPU", run_idx, show_all_elements, max_elements)
    
    if npu_exists and not cpu_exists:
        print("\nℹ️  Only NPU output available. Showing NPU output only.")
        return show_single_output(NPU_ALL_OUTPUTS_FILE, "NPU", run_idx, show_all_elements, max_elements)
    
    # Both exist - do full comparison
    # Load data
    cpu_data = np.load(CPU_ALL_OUTPUTS_FILE)
    npu_data = np.load(NPU_ALL_OUTPUTS_FILE)
    
    cpu_outputs = cpu_data['outputs']
    npu_outputs = npu_data['outputs']
    cpu_inputs = cpu_data.get('inputs', None)
    
    num_runs = min(len(cpu_outputs), len(npu_outputs))
    
    if run_idx >= num_runs:
        print(f"ERROR: Run index {run_idx} out of range (total runs: {num_runs})")
        return False
    
    print(f"\nTotal runs available: {num_runs}")
    print(f"Comparing run: {run_idx + 1}")
    print(f"Output shape: {cpu_outputs[run_idx].shape}")
    print(f"Expected shape: ({CONFIG_BATCH_SIZE}, {CONFIG_NUM_TASKS})")
    
    # Get specific run data
    cpu_output = cpu_outputs[run_idx]
    npu_output = npu_outputs[run_idx]
    cpu_input = cpu_inputs[run_idx] if cpu_inputs is not None else None
    
    # Print input info
    if cpu_input is not None:
        print("\n" + PRINT_SEPARATOR)
        print("Input Data Statistics")
        print(PRINT_SEPARATOR)
        print(f"  Shape: {cpu_input.shape}")
        print(f"  Min: {cpu_input.min():.6f}")
        print(f"  Max: {cpu_input.max():.6f}")
        print(f"  Mean: {cpu_input.mean():.6f}")
        print(f"  Std: {cpu_input.std():.6f}")
    
    # Element-wise comparison
    diff = np.abs(cpu_output - npu_output)
    rel_diff = diff / (np.abs(cpu_output) + 1e-8)
    matches = diff <= COMPARISON_TOLERANCE
    match_count = np.sum(matches)
    total_count = matches.size
    
    print("\n" + PRINT_SEPARATOR)
    print("Overall Statistics")
    print(PRINT_SEPARATOR)
    print(f"  Total elements: {total_count}")
    print(f"  Matching elements: {match_count} ({100*match_count/total_count:.2f}%)")
    print(f"  Non-matching: {total_count - match_count}")
    print(f"  Tolerance: {COMPARISON_TOLERANCE}")
    print(f"\n  Max absolute difference: {diff.max():.6e}")
    print(f"  Mean absolute difference: {diff.mean():.6e}")
    print(f"  Max relative difference: {rel_diff.max():.6e}")
    print(f"  Mean relative difference: {rel_diff.mean():.6e}")
    
    # Show outputs
    print("\n" + "=" * 80)
    print("CPU Output")
    print("=" * 80)
    print(format_array(cpu_output, "CPU", width=12, precision=6))
    
    print("\n" + "=" * 80)
    print("NPU Output")
    print("=" * 80)
    print(format_array(npu_output, "NPU", width=12, precision=6))
    
    print("\n" + "=" * 80)
    print("Absolute Difference (|CPU - NPU|)")
    print("=" * 80)
    print(format_array(diff, "Diff", width=12, precision=6))
    
    # Side-by-side comparison for first N elements
    elements_to_show = total_count if show_all_elements else min(max_elements, total_count)
    
    print("\n" + "=" * 80)
    print(f"Side-by-Side Comparison (first {elements_to_show} elements)")
    print("=" * 80)
    print(f"{'Index':>6} | {'CPU':>14} | {'NPU':>14} | {'Abs Diff':>14} | {'Rel Diff':>14} | {'Match':>6}")
    print("-" * 80)
    
    cpu_flat = cpu_output.flatten()
    npu_flat = npu_output.flatten()
    diff_flat = diff.flatten()
    rel_diff_flat = rel_diff.flatten()
    
    for i in range(elements_to_show):
        match = "YES" if diff_flat[i] <= COMPARISON_TOLERANCE else "NO"
        print(f"{i:>6} | {cpu_flat[i]:>14.6e} | {npu_flat[i]:>14.6e} | "
              f"{diff_flat[i]:>14.6e} | {rel_diff_flat[i]:>14.6e} | {match:>6}")
    
    if not show_all_elements and elements_to_show < total_count:
        print(f"\n... ({total_count - elements_to_show} more elements)")
    
    # Final result
    all_match = match_count == total_count
    print("\n" + "=" * 80)
    if all_match:
        print("RESULT: All elements match within tolerance")
    else:
        print(f"RESULT: {total_count - match_count} elements exceed tolerance")
    print("=" * 80)
    
    return all_match


def compare_all_runs():
    """Compare all available runs."""
    print("\n" + "=" * 80)
    print("Comparing All Available Runs")
    print("=" * 80)
    
    cpu_exists = os.path.exists(CPU_ALL_OUTPUTS_FILE)
    npu_exists = os.path.exists(NPU_ALL_OUTPUTS_FILE)
    
    if not cpu_exists:
        print(f"\n⚠️  WARNING: CPU output file not found: {CPU_ALL_OUTPUTS_FILE}")
        print("    Skipping CPU in comparison.")
    
    if not npu_exists:
        print(f"\n⚠️  WARNING: NPU output file not found: {NPU_ALL_OUTPUTS_FILE}")
        print("    Skipping NPU in comparison.")
    
    if not cpu_exists and not npu_exists:
        print("\n❌ ERROR: No output files found. Please run 'python main.py' first.")
        return False
    
    if not cpu_exists or not npu_exists:
        print("\nℹ️  Cannot compare - need both CPU and NPU outputs.")
        print("    Run 'python main.py' to generate both outputs.")
        return False
    
    cpu_data = np.load(CPU_ALL_OUTPUTS_FILE)
    npu_data = np.load(NPU_ALL_OUTPUTS_FILE)
    
    cpu_outputs = cpu_data['outputs']
    npu_outputs = npu_data['outputs']
    
    num_runs = min(len(cpu_outputs), len(npu_outputs))
    
    print(f"\nTotal runs to compare: {num_runs}")
    
    all_passed = True
    results = []
    
    for run_idx in range(num_runs):
        cpu_output = cpu_outputs[run_idx]
        npu_output = npu_outputs[run_idx]
        
        diff = np.abs(cpu_output - npu_output)
        match_count = np.sum(diff <= COMPARISON_TOLERANCE)
        total_count = diff.size
        max_diff = diff.max()
        
        passed = match_count == total_count
        all_passed = all_passed and passed
        
        results.append({
            'run': run_idx + 1,
            'passed': passed,
            'matches': match_count,
            'total': total_count,
            'max_diff': max_diff
        })
    
    # Print summary table
    print("\n" + "-" * 80)
    print(f"{'Run':>6} | {'Status':>8} | {'Matches':>12} | {'Max Diff':>14}")
    print("-" * 80)
    
    for r in results:
        status = "PASS" if r['passed'] else "FAIL"
        match_str = f"{r['matches']}/{r['total']}"
        print(f"{r['run']:>6} | {status:>8} | {match_str:>12} | {r['max_diff']:>14.6e}")
    
    print("-" * 80)
    
    if all_passed:
        print(f"\nAll {num_runs} runs PASSED")
    else:
        failed_count = sum(1 for r in results if not r['passed'])
        print(f"\n{failed_count}/{num_runs} runs FAILED")
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(description='Show full CPU vs NPU comparison')
    parser.add_argument('--run', type=int, default=0,
                        help='Run index to compare (0-based, default: 0)')
    parser.add_argument('--all', action='store_true',
                        help='Show all elements (may be large)')
    parser.add_argument('--max', type=int, default=1024,
                        help='Maximum elements to show (default: 1024)')
    parser.add_argument('--summary', action='store_true',
                        help='Show summary of all runs only')
    parser.add_argument('--cpu-only', action='store_true',
                        help='Show only CPU output')
    parser.add_argument('--npu-only', action='store_true',
                        help='Show only NPU output')
    
    args = parser.parse_args()
    
    # Determine what to show
    show_cpu = not args.npu_only
    show_npu = not args.cpu_only
    
    if args.summary:
        success = compare_all_runs()
    else:
        success = show_full_comparison(
            run_idx=args.run,
            show_all_elements=args.all,
            max_elements=args.max,
            show_cpu=show_cpu,
            show_npu=show_npu
        )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
