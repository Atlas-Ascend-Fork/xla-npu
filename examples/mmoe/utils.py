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
MMOE Test Utilities - Common functions for CPU and NPU tests
"""

import os
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp

# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    CONFIG_BATCH_SIZE,
    CONFIG_INPUT_DIM,
    CONFIG_NUM_TASKS,
    COMPARISON_TOLERANCE,
    PRINT_SEPARATOR,
    OUTPUT_SAMPLE_SIZE,
)
from init_params import (
    setup_model_and_params,
)

from common.utils import (
    ensure_dir,
    save_results,
    generate_input,
    print_input_stats,
    print_results,
)


# ============================================================================
# Input Generation
# ============================================================================

def generate_mmoe_input(seed=None):
    """Generate random input data suitable for MMOE model.
    
    The input data follows a normal distribution with mean 0 and std 1,
    which is suitable for neural network inference after typical feature normalization.
    
    Args:
        seed: Random seed for reproducibility. If None, uses random seed.
    
    Returns:
        jnp.ndarray: Input tensor with shape (CONFIG_BATCH_SIZE, CONFIG_INPUT_DIM)
    """
    x = generate_input((CONFIG_BATCH_SIZE, CONFIG_INPUT_DIM), seed)
    return jnp.array(x)


# ============================================================================
# JIT Compilation and Inference
# ============================================================================

# Module-level JIT cache
_jit_cache = {}


def get_jitted_forward(model):
    """Get or create JIT-compiled forward function for the model.
    
    Args:
        model: MMOE model instance
        
    Returns:
        Callable: JIT-compiled forward function
    """
    model_id = id(model)
    if model_id not in _jit_cache:
        @jax.jit
        def jitted_forward(params, x):
            return model.apply(params, x, train=False)
        _jit_cache[model_id] = jitted_forward
    return _jit_cache[model_id]


def run_inference(model, params, x):
    """Run inference and return results.
    
    Args:
        model: MMOE model instance
        params: Model parameters
        x: Input tensor
        
    Returns:
        np.ndarray: Output as numpy array
    """
    jitted_forward = get_jitted_forward(model)
    output = jitted_forward(params, x)
    return np.array(output)


def time_inference(model, params, x):
    """Run inference and measure execution time.
    
    Args:
        model: MMOE model instance
        params: Model parameters
        x: Input tensor
        
    Returns:
        tuple: (output, execution_time_ms)
    """
    
    # Measure actual execution time
    start_time = time.time()
    output = run_inference(model, params, x)
    end_time = time.time()
    
    execution_time_ms = (end_time - start_time) * 1000
    return output, execution_time_ms


def run_inference_and_save(model, params, x, device_name, output_file):
    """Run inference and save results to file.
    
    Args:
        model: MMOE model instance
        params: Model parameters
        x: Input tensor
        device_name: Device name ('CPU' or 'NPU')
        output_file: Path to save output
        
    Returns:
        tuple: (output, execution_time_ms)
    """
    output, execution_time_ms = time_inference(model, params, x)
    save_results(output_file, output=output, input=x, execution_time=execution_time_ms)
    return output, execution_time_ms
