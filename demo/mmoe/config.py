# -*- coding:utf-8 -*-
"""
MMOE Configuration - All constants and configuration parameters
"""

# ============================================================================
# Model Architecture Configuration
# ============================================================================

# Input and batch size
CONFIG_INPUT_DIM = 256         # Input feature dimension
CONFIG_BATCH_SIZE = 64         # Batch size for inference

# Model structure
CONFIG_NUM_EXPERTS = 2         # Number of expert networks
CONFIG_NUM_TASKS = 2           # Number of tasks (must match CONFIG_TASK_TYPES length)

# DNN hidden units
CONFIG_EXPERT_DNN_HIDDEN_UNITS = (256, 128)   # Expert DNN hidden layers
CONFIG_GATE_DNN_HIDDEN_UNITS = (128,)          # Gate DNN hidden layers
CONFIG_TOWER_DNN_HIDDEN_UNITS = (128,)         # Tower DNN hidden layers

# DNN settings
CONFIG_DNN_ACTIVATION = 'relu'               # Activation function
CONFIG_TASK_TYPES = ('binary', 'binary')     # Task types (length must equal CONFIG_NUM_TASKS)

# ============================================================================
# File Paths
# ============================================================================

CONFIG_MODEL_PARAMS_FILE = "model_params.npz"  # Model parameters file name

# ============================================================================
# Test Configuration
# ============================================================================

# Inference settings
DEFAULT_NUM_RUNS = 1          # Default number of runs per test
DEFAULT_NUM_TESTS = 1         # Default number of test iterations

# Comparison settings
COMPARISON_TOLERANCE = 1e-4   # Tolerance for CPU/NPU result comparison

# ============================================================================
# JAX/NPU Configuration
# ============================================================================

# Platform settings
NPU_PLATFORM_NAME = 'npu'
NPU_LIBRARY_NAME = 'pjrt_c_api_npu_plugin.so'

# Environment variables
ENV_PJRT_NAMES_AND_LIBRARY_PATHS = 'PJRT_NAMES_AND_LIBRARY_PATHS'
ENV_JAX_PLATFORMS = 'JAX_PLATFORMS'

# ============================================================================
# Temporary File Paths
# ============================================================================

# Output file paths
CPU_OUTPUT_FILE = "/tmp/mmoe_cpu_output.npz"
NPU_OUTPUT_FILE = "/tmp/mmoe_npu_output.npz"
CPU_ALL_OUTPUTS_FILE = "/tmp/mmoe_cpu_all_outputs.npz"
NPU_ALL_OUTPUTS_FILE = "/tmp/mmoe_npu_all_outputs.npz"
INPUT_DATA_FILE = "/tmp/mmoe_input_data.npz"

# ============================================================================
# Print Settings
# ============================================================================

PRINT_SEPARATOR = "=" * 60
OUTPUT_SAMPLE_SIZE = 5        # Number of samples to print
