import os
import sys
import numpy as np
import jax

# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.utils import (
    setup_device_environment,
    time_function,
    save_results,
    compare_results,
)
from common.config import (
    PRINT_SEPARATOR,
)
from config import (
    OPS_OUTPUT_DIR,
    DEFAULT_NUM_RUNS,
)


def run_operation(op_config, num_runs=1, device='CPU', inputs_list=None, run_number=1):
    """Run a single operation multiple times on specified device.
    
    Args:
        op_config: Operation configuration
        num_runs: Number of runs to execute
        device: Device to run on ('CPU' or 'NPU')
        inputs_list: Pre-generated inputs list. If None, generate new inputs.
        run_number: Run number for output directory structure
    
    Returns:
        tuple: (outputs, inputs, execution_times)
    """
    print(f"{'=' * 60}")
    print(f"Running {op_config['name']} on {device}")
    print(f"{'=' * 60}")
    
    outputs = []
    actual_inputs_list = []
    execution_times = []
    
    # Set device
    setup_device_environment(device)
    
    for i in range(num_runs):
        # Use pre-generated inputs if provided, otherwise generate new ones
        if inputs_list is not None and i < len(inputs_list):
            inputs = inputs_list[i]
        else:
            inputs = op_config['inputs']()
        actual_inputs_list.append(inputs)
        
        # Print input shapes for first run
        if i == 0:
            print("Input shapes:")
            for name, value in inputs.items():
                print(f"  {name}: {value.shape}")
            print()
        
        # Create jitted function
        func = jax.jit(op_config['func'])
        
        # Execute and measure time
        output, exec_time = time_function(func, **inputs)
        outputs.append(output)
        execution_times.append(exec_time)
        
        print(f"[{device}] Run {i+1}/{num_runs}: {exec_time:.2f} ms")
    
    # Calculate statistics
    avg_time = np.mean(execution_times)
    
    print(f"\n[{device}] Average execution time: {avg_time:.2f} ms")
    print()
    
    return outputs, actual_inputs_list, execution_times



