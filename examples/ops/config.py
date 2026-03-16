# -*- coding:utf-8 -*-
"""
Operations Configuration - All constants and configuration parameters
"""

import os
import sys
import numpy as np
import jax.numpy as jnp

from common.config import (
    COMPARISON_TOLERANCE,
    PRINT_SEPARATOR,
    OUTPUT_SAMPLE_SIZE,
    OUTPUT_DIR,
)
from common.utils import create_output_dirs, get_output_files

from common.utils import generate_input

# ============================================================================
# Operation Configuration
# ============================================================================

# Input shape configuration
CONFIG_INPUT_SHAPE = (2, 3)        # Default input shape for operations
CONFIG_REDUCE_SHAPE = (3, 3)       # Input shape for reduce operations
CONFIG_DOT_SHAPE_1 = (2, 3)        # First input shape for dot operation
CONFIG_DOT_SHAPE_2 = (3, 2)        # Second input shape for dot operation

# ============================================================================
# Test Configuration
# ============================================================================

# Test description
DESCRIPTION = "JAX Operations Comparison"

# Inference settings
DEFAULT_NUM_RUNS = 1          # Default number of runs per test

# ============================================================================
# Temporary File Paths
# ============================================================================

# Output directories
OPS_OUTPUT_DIR, _ = create_output_dirs(OUTPUT_DIR, "ops")

# Output file paths (default without run number)
output_files = get_output_files(OPS_OUTPUT_DIR)
CPU_ALL_OUTPUTS_FILE = output_files['cpu_all_outputs_file']
NPU_ALL_OUTPUTS_FILE = output_files['npu_all_outputs_file']
INPUT_DATA_FILE = output_files['input_data_file']

# ============================================================================
# Operation Configuration List
# ============================================================================

# Operation configuration list
ops_configs = [
    {
        'name': 'Add Operation',
        'inputs': lambda: {
            'x1': generate_input(CONFIG_INPUT_SHAPE),
            'x2': generate_input(CONFIG_INPUT_SHAPE)
        },
        'func': lambda x1, x2: x1 + x2
    },
    {
        'name': 'Tanh Operation',
        'inputs': lambda: {
            'x': generate_input(CONFIG_INPUT_SHAPE)
        },
        'func': lambda x: jnp.tanh(x)
    },
    {
        'name': 'Multiply Operation',
        'inputs': lambda: {
            'x1': generate_input(CONFIG_INPUT_SHAPE),
            'x2': generate_input(CONFIG_INPUT_SHAPE)
        },
        'func': lambda x1, x2: x1 * x2
    },
    {
        'name': 'Dot Operation',
        'inputs': lambda: {
            'x1': generate_input(CONFIG_DOT_SHAPE_1),
            'x2': generate_input(CONFIG_DOT_SHAPE_2)
        },
        'func': lambda x1, x2: jnp.dot(x1, x2)
    },
    {
        'name': 'Divide Operation',
        'inputs': lambda: {
            'x1': generate_input(CONFIG_INPUT_SHAPE) * 10,
            'x2': generate_input(CONFIG_INPUT_SHAPE) + 1  # Avoid division by zero
        },
        'func': lambda x1, x2: x1 / x2
    },
    {
        'name': 'Subtract Operation',
        'inputs': lambda: {
            'x1': generate_input(CONFIG_INPUT_SHAPE) * 10,
            'x2': generate_input(CONFIG_INPUT_SHAPE)
        },
        'func': lambda x1, x2: x1 - x2
    },
    {
        'name': 'Negate Operation',
        'inputs': lambda: {
            'x': generate_input(CONFIG_INPUT_SHAPE)
        },
        'func': lambda x: -x
    },
    {
        'name': 'Exp Operation',
        'inputs': lambda: {
            'x': generate_input(CONFIG_INPUT_SHAPE)
        },
        'func': lambda x: jnp.exp(x)
    },
    {
        'name': 'Maximum Operation',
        'inputs': lambda: {
            'x1': generate_input(CONFIG_INPUT_SHAPE),
            'x2': generate_input(CONFIG_INPUT_SHAPE)
        },
        'func': lambda x1, x2: jnp.maximum(x1, x2)
    },
    {
        'name': 'Concatenate Operation',
        'inputs': lambda: {
            'x1': generate_input(CONFIG_INPUT_SHAPE),
            'x2': generate_input(CONFIG_INPUT_SHAPE)
        },
        'func': lambda x1, x2: jnp.concatenate([x1, x2], axis=0)
    },
    {
        'name': 'ReduceMax Operation',
        'inputs': lambda: {
            'x': generate_input(CONFIG_REDUCE_SHAPE)
        },
        'func': lambda x: jnp.max(x, axis=0)
    },
    {
        'name': 'ReduceSum Operation',
        'inputs': lambda: {
            'x': generate_input(CONFIG_REDUCE_SHAPE)
        },
        'func': lambda x: jnp.sum(x, axis=0)
    }
]
