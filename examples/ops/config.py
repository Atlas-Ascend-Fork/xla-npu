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
Operations Configuration - All constants and configuration parameters
"""

import os
import sys
import numpy as np
import jax
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
CPU_OUTPUT_FILE = output_files['cpu_output_file']
NPU_OUTPUT_FILE = output_files['npu_output_file']
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
    },
    {
        'name': 'Rsqrt Operation',
        'inputs': lambda: {
            'x': np.abs(generate_input(CONFIG_INPUT_SHAPE)) + 0.1  # Ensure all positive
        },
        'func': lambda x: jax.lax.rsqrt(x)
    },
    {
        'name': 'Sin Operation',
        'inputs': lambda: {
            'x': generate_input(CONFIG_INPUT_SHAPE)
        },
        'func': lambda x: jnp.sin(x)
    },
    {
        'name': 'Cos Operation',
        'inputs': lambda: {
            'x': generate_input(CONFIG_INPUT_SHAPE)
        },
        'func': lambda x: jnp.cos(x)
    },
    {
        'name': 'Transpose Operation',
        'inputs': lambda: {
            'x': generate_input((1, 1, 64))
        },
        'func': lambda x: jnp.transpose(x, (0, 2, 1))
    },
    {
        'name': 'Slice Operation',
        'inputs': lambda: {
            'x': generate_input((4, 4))
        },
        'func': lambda x: x[1:3, 0:2]
    },
    {
        'name': 'Gather Operation',
        'inputs': lambda: {
            'x': generate_input((4, 4))
        },
        'func': lambda x: x[jnp.array([0, 2]), :]
    },
    {
        'name': 'Compare EQ Operation',
        'inputs': lambda: {
            'x1': generate_input(CONFIG_INPUT_SHAPE),
            'x2': generate_input(CONFIG_INPUT_SHAPE)
        },
        'func': lambda x1, x2: x1 == x2
    },
    {
        'name': 'Select Operation',
        'inputs': lambda: {
            'x1': generate_input(CONFIG_INPUT_SHAPE),
            'x2': generate_input(CONFIG_INPUT_SHAPE)
        },
        'func': lambda x1, x2: jnp.where(x1 > 0, x1, x2)
    },
    {
        'name': 'And Operation',
        'inputs': lambda: {
            'x1': generate_input(CONFIG_INPUT_SHAPE) > 0,
            'x2': generate_input(CONFIG_INPUT_SHAPE) > 0
        },
        'func': lambda x1, x2: jnp.logical_and(x1, x2)
    }
]
