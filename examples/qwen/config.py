#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""
Qwen2 Configuration - All constants and configuration parameters
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

# Model structure
CONFIG_HIDDEN_SIZE = 896
CONFIG_NUM_HIDDEN_LAYERS = 24
CONFIG_NUM_ATTENTION_HEADS = 14
CONFIG_NUM_KEY_VALUE_HEADS = 2
CONFIG_INTERMEDIATE_SIZE = 4864
CONFIG_MAX_POSITION_EMBEDDINGS = 32768
CONFIG_RMS_NORM_EPS = 1e-6
CONFIG_ROPE_THETA = 1000000.0
CONFIG_VOCAB_SIZE = 151936

# Inference settings
CONFIG_BATCH_SIZE = 1
CONFIG_SEQ_LEN = 8

# ============================================================================
# File Paths
# ============================================================================

CONFIG_MODEL_PARAMS_FILE = "model_params.npz"  # Model parameters file name

# ============================================================================
# Test Configuration
# ============================================================================

# Test description
DESCRIPTION = "Qwen2 JAX Implementation"

# Inference settings
DEFAULT_NUM_RUNS = 1          # Default number of runs per test

# ============================================================================
# Temporary File Paths
# ============================================================================

# Output directories
QWEN_OUTPUT_DIR, TEST_OUTPUT_DIR = create_output_dirs(OUTPUT_DIR, "qwen")

# Output file paths (default without run number)
output_files = get_output_files(QWEN_OUTPUT_DIR, TEST_OUTPUT_DIR)
CPU_OUTPUT_FILE = output_files['cpu_output_file']
NPU_OUTPUT_FILE = output_files['npu_output_file']
CPU_ALL_OUTPUTS_FILE = output_files['cpu_all_outputs_file']
NPU_ALL_OUTPUTS_FILE = output_files['npu_all_outputs_file']
INPUT_DATA_FILE = output_files['input_data_file']
