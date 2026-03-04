# -*- coding:utf-8 -*-
"""
MMOE NPU Test - Run inference on NPU
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_npu_test():
    """Run NPU inference test."""
    print("\n" + "=" * 60)
    print("Running NPU inference...")
    print("=" * 60)

    PLATFORM_NAME = 'npu'
    LIBRARY_PATH = os.path.abspath("pjrt_c_api_npu_plugin.so")
    os.environ['PJRT_NAMES_AND_LIBRARY_PATHS'] = f"{PLATFORM_NAME}:{LIBRARY_PATH}"
    os.environ['JAX_PLATFORMS'] = PLATFORM_NAME

    import importlib
    import jax
    importlib.reload(jax)

    print(f"JAX backend: {jax.devices()}")
    print(f"JAX default backend: {jax.default_backend()}")

    from model import (
        setup_model_and_data,
        run_inference_and_save,
        CONFIG_BATCH_SIZE,
        CONFIG_NUM_TASKS,
    )

    model, params, x = setup_model_and_data()
    output = run_inference_and_save(model, params, x, "NPU", "/tmp/mmoe_npu_output.npz")
    return output, x


if __name__ == "__main__":
    output, x = run_npu_test()
    print("\nNPU test completed successfully!")
