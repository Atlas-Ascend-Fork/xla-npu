# Platform settings
NPU_PLATFORM_NAME = 'npu'
NPU_LIBRARY_NAME = 'pjrt_c_api_npu_plugin.so'

# Environment variables
ENV_PJRT_NAMES_AND_LIBRARY_PATHS = 'PJRT_NAMES_AND_LIBRARY_PATHS'
ENV_JAX_PLATFORMS = 'JAX_PLATFORMS'

# Comparison settings
COMPARISON_TOLERANCE = 1e-4   # Tolerance for CPU/NPU result comparison

# Print settings
PRINT_SEPARATOR = "=" * 60
OUTPUT_SAMPLE_SIZE = 5        # Number of samples to print

# Output directories
import os
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
