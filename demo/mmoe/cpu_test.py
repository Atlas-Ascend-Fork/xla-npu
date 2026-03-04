# -*- coding:utf-8 -*-
"""
MMOE CPU Test - Run inference on CPU
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import (
    setup_model_and_data,
    run_inference_and_save,
    CONFIG_BATCH_SIZE,
    CONFIG_NUM_TASKS,
)


def run_cpu_test():
    """Run CPU inference test."""
    print("=" * 60)
    print("Running CPU inference...")
    print("=" * 60)
    model, params, x = setup_model_and_data()
    output = run_inference_and_save(model, params, x, "CPU", "/tmp/mmoe_cpu_output.npz")
    return output, x


if __name__ == "__main__":
    output, x = run_cpu_test()
    print("\nCPU test completed successfully!")
