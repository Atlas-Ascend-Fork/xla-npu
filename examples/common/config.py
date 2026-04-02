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
