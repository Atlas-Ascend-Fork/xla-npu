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
MMOE Configuration - All constants and configuration parameters
"""

import os
import sys

from common.config import (
    COMPARISON_TOLERANCE,
    PRINT_SEPARATOR,
    OUTPUT_SAMPLE_SIZE,
    OUTPUT_DIR,
)
from common.utils import create_output_dirs, get_output_files

# ============================================================================
# Model Architecture Configuration
# ============================================================================

# Input and batch size
CONFIG_INPUT_DIM = 1080         # Input feature dimension
CONFIG_BATCH_SIZE = 320         # Batch size for inference

# Model structure
CONFIG_NUM_EXPERTS = 4         # Number of expert networks (scaled 2x)
CONFIG_NUM_TASKS = 4           # Number of tasks (must match CONFIG_TASK_TYPES length, scaled 2x)

# DNN hidden units (scaled down proportionally with input dim)
CONFIG_EXPERT_DNN_HIDDEN_UNITS = (64, 32)     # Expert DNN hidden layers
CONFIG_GATE_DNN_HIDDEN_UNITS = (32,)           # Gate DNN hidden layers
CONFIG_TOWER_DNN_HIDDEN_UNITS = (32,)          # Tower DNN hidden layers

# DNN settings
CONFIG_DNN_ACTIVATION = 'relu'               # Activation function
CONFIG_TASK_TYPES = ('binary', 'binary', 'binary', 'binary')  # Task types (length must equal CONFIG_NUM_TASKS, scaled 2x)

# ============================================================================
# File Paths
# ============================================================================

CONFIG_MODEL_PARAMS_FILE = "model_params.npz"  # Model parameters file name

# ============================================================================
# Test Configuration
# ============================================================================

# Test description
DESCRIPTION = "MMOE JAX Implementation"

# Inference settings
DEFAULT_NUM_RUNS = 1          # Default number of runs per test

# ============================================================================
# Temporary File Paths
# ============================================================================

# Output directories
MMOE_OUTPUT_DIR, TEST_OUTPUT_DIR = create_output_dirs(OUTPUT_DIR, "mmoe")

# Output file paths (default without run number)
output_files = get_output_files(MMOE_OUTPUT_DIR, TEST_OUTPUT_DIR)
CPU_OUTPUT_FILE = output_files['cpu_output_file']
NPU_OUTPUT_FILE = output_files['npu_output_file']
CPU_ALL_OUTPUTS_FILE = output_files['cpu_all_outputs_file']
NPU_ALL_OUTPUTS_FILE = output_files['npu_all_outputs_file']
INPUT_DATA_FILE = output_files['input_data_file']
