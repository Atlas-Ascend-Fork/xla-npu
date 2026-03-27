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

import os

# Platform settings
NPU_PLATFORM_NAME = 'npu'
NPU_LIBRARY_NAME = os.path.join(
    os.environ.get('XLA_NPU_ROOT_DIR', ''),
    'build/code/xla/bazel-bin/xla/xla_npu/pjrt/c/pjrt_c_api_npu_plugin.so'
)

# Environment variables
ENV_PJRT_NAMES_AND_LIBRARY_PATHS = 'PJRT_NAMES_AND_LIBRARY_PATHS'
ENV_JAX_PLATFORMS = 'JAX_PLATFORMS'

# Comparison settings
COMPARISON_TOLERANCE = 1e-4   # Tolerance for CPU/NPU result comparison
MAX_VALUES_TO_PRINT = 36      # Maximum number of values to print when comparison fails

# Print settings
PRINT_SEPARATOR = "=" * 60
OUTPUT_SAMPLE_SIZE = 5        # Number of samples to print

# Output directories
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
