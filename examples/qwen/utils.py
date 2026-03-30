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
Qwen2 Test Utilities - Common functions for CPU and NPU tests
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
    CONFIG_SEQ_LEN,
    CONFIG_VOCAB_SIZE,
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

def generate_qwen_input(seed=None):
    """Generate random input data suitable for Qwen2 model.
    
    Args:
        seed: Random seed for reproducibility. If None, uses random seed.
    
    Returns:
        jnp.ndarray: Input tensor with shape (CONFIG_BATCH_SIZE, CONFIG_SEQ_LEN)
    """
    if seed is None:
        seed = int(np.random.randint(0, 10000))
    
    rng = np.random.RandomState(seed)
    input_ids = rng.randint(0, CONFIG_VOCAB_SIZE, size=(CONFIG_BATCH_SIZE, CONFIG_SEQ_LEN))
    return jnp.array(input_ids)


def create_fixed_input(batch_size, seq_len, vocab_size):
    """Create fixed input data for reproducible testing.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        vocab_size: Vocabulary size

    Returns:
        jnp.ndarray: Fixed input tensor with shape (batch_size, seq_len)
    """
    # Use fixed seed for reproducibility
    rng = np.random.RandomState(42)
    input_ids = rng.randint(0, vocab_size, size=(batch_size, seq_len))
    return jnp.array(input_ids)


# ============================================================================
# JIT Compilation and Inference
# ============================================================================

# Module-level JIT cache
_jit_cache = {}


def get_jitted_forward(model):
    """Get or create JIT-compiled forward function for the model.
    
    Args:
        model: Qwen2 model instance
        
    Returns:
        Callable: JIT-compiled forward function
    """
    model_id = id(model)
    if model_id not in _jit_cache:
        @jax.jit
        def jitted_forward(params, input_ids, attention_mask):
            return model.apply(params, input_ids, attention_mask=attention_mask, deterministic=False)
        _jit_cache[model_id] = jitted_forward
    return _jit_cache[model_id]


def run_inference(model, params, input_ids, attention_mask):
    """Run inference and return results.
    
    Args:
        model: Qwen2 model instance
        params: Model parameters
        input_ids: Input tensor
        attention_mask: Attention mask
        
    Returns:
        np.ndarray: Output as numpy array
    """
    jitted_forward = get_jitted_forward(model)
    output = jitted_forward(params, input_ids, attention_mask)
    return np.array(output)


def time_inference(model, params, input_ids, attention_mask):
    """Run inference and measure execution time.
    
    Args:
        model: Qwen2 model instance
        params: Model parameters
        input_ids: Input tensor
        attention_mask: Attention mask
        
    Returns:
        tuple: (output, execution_time_ms)
    """
    # Measure actual execution time
    start_time = time.time()
    output = run_inference(model, params, input_ids, attention_mask)
    end_time = time.time()
    
    execution_time_ms = (end_time - start_time) * 1000
    return output, execution_time_ms


def run_inference_and_save(model, params, input_ids, attention_mask, device_name, output_file):
    """Run inference and save results to file.
    
    Args:
        model: Qwen2 model instance
        params: Model parameters
        input_ids: Input tensor
        attention_mask: Attention mask
        device_name: Device name ('CPU' or 'NPU')
        output_file: Path to save output
        
    Returns:
        tuple: (output, execution_time_ms)
    """
    output, execution_time_ms = time_inference(model, params, input_ids, attention_mask)
    save_results(output_file, output=output, input=input_ids, execution_time=execution_time_ms)
    return output, execution_time_ms
