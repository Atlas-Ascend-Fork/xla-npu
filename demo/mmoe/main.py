# -*- coding:utf-8 -*-
"""
MMOE Test - Main entry point for CPU and NPU comparison test
"""

import os
import sys
import subprocess
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    """Run CPU and NPU tests and compare results."""
    print("MMOE JAX Implementation Test")
    print("=" * 50)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    result_cpu = subprocess.run(
        [sys.executable, os.path.join(script_dir, "cpu_test.py")],
        capture_output=False
    )

    result_npu = subprocess.run(
        [sys.executable, os.path.join(script_dir, "npu_test.py")],
        capture_output=False
    )

    from model import compare_results

    cpu_output_path = "/tmp/mmoe_cpu_output.npz"
    npu_output_path = "/tmp/mmoe_npu_output.npz"

    if not os.path.exists(cpu_output_path):
        print(f"WARNING: CPU output file not found: {cpu_output_path}")
        sys.exit(1)

    if not os.path.exists(npu_output_path):
        print(f"WARNING: NPU output file not found: {npu_output_path}")
        sys.exit(1)

    cpu_data = np.load(cpu_output_path)
    npu_data = np.load(npu_output_path)

    compare_results(cpu_data['output'], npu_data['output'], cpu_data['input'], npu_data['input'])

    if os.path.exists("/tmp/mmoe_cpu_output.npz"):
        os.remove("/tmp/mmoe_cpu_output.npz")
    if os.path.exists("/tmp/mmoe_npu_output.npz"):
        os.remove("/tmp/mmoe_npu_output.npz")


if __name__ == "__main__":
    main()
