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
Initialize Qwen2 model parameters on CPU and save to file
"""

import os
import sys

# Force CPU environment before importing jax
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp
import numpy as np

# Import model definition
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import create_qwen_model, create_causal_mask
from utils import create_fixed_input
from config import (
    CONFIG_HIDDEN_SIZE,
    CONFIG_NUM_HIDDEN_LAYERS,
    CONFIG_NUM_ATTENTION_HEADS,
    CONFIG_NUM_KEY_VALUE_HEADS,
    CONFIG_INTERMEDIATE_SIZE,
    CONFIG_MAX_POSITION_EMBEDDINGS,
    CONFIG_VOCAB_SIZE,
    CONFIG_BATCH_SIZE,
    CONFIG_SEQ_LEN,
    CONFIG_MODEL_PARAMS_FILE,
)


def save_params_with_npz(params, filepath: str):
    """Save model parameters to npz file.
    
    Args:
        params: Model parameters (Flax params dict)
        filepath: Path to save parameters
    """
    params_np = jax.tree_util.tree_map(lambda x: np.array(x), params)
    
    def flatten_dict(d, parent_key='', sep='.'):
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(flatten_dict(v, new_key, sep=sep))
            else:
                items[new_key] = v
        return items
    
    flat_params = flatten_dict(params_np)
    np.savez(filepath, **flat_params)
    print(f"Parameters saved to: {filepath}")


def load_params_with_npz(filepath: str):
    """Load model parameters from npz file.
    
    Args:
        filepath: Path to load parameters from
        
    Returns:
        dict: Model parameters (Flax params dict)
    """
    data = np.load(filepath)
    
    def unflatten_dict(flat_dict, sep='.'):
        result = {}
        for key, value in flat_dict.items():
            parts = key.split(sep)
            current = result
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = np.array(value)
        return result
    
    params_jax = unflatten_dict(dict(data))
    return params_jax


def get_model_params_path():
    """Get the path to model parameters file."""
    return os.path.join(os.path.dirname(__file__), CONFIG_MODEL_PARAMS_FILE)


def setup_model_and_params():
    """Setup model and load parameters.
    
    Returns:
        tuple: (model, params)
            - model: Qwen2 model instance
            - params: Model parameters
    """
    model = create_qwen_model(
        hidden_size=CONFIG_HIDDEN_SIZE,
        num_hidden_layers=CONFIG_NUM_HIDDEN_LAYERS,
        num_attention_heads=CONFIG_NUM_ATTENTION_HEADS,
        num_key_value_heads=CONFIG_NUM_KEY_VALUE_HEADS,
        intermediate_size=CONFIG_INTERMEDIATE_SIZE,
        max_position_embeddings=CONFIG_MAX_POSITION_EMBEDDINGS,
        vocab_size=CONFIG_VOCAB_SIZE,
    )
    
    save_filepath = get_model_params_path()
    params = load_params_with_npz(save_filepath)
    
    return model, params


def main():
    """Initialize Qwen2 model parameters on CPU and save to file."""
    print("Initializing Qwen2 model parameters on CPU...")
    
    print(f"Hidden size: {CONFIG_HIDDEN_SIZE}")
    print(f"Num hidden layers: {CONFIG_NUM_HIDDEN_LAYERS}")
    print(f"Num attention heads: {CONFIG_NUM_ATTENTION_HEADS}")
    print(f"Vocab size: {CONFIG_VOCAB_SIZE}")

    # Create model
    model = create_qwen_model(
        hidden_size=CONFIG_HIDDEN_SIZE,
        num_hidden_layers=CONFIG_NUM_HIDDEN_LAYERS,
        num_attention_heads=CONFIG_NUM_ATTENTION_HEADS,
        num_key_value_heads=CONFIG_NUM_KEY_VALUE_HEADS,
        intermediate_size=CONFIG_INTERMEDIATE_SIZE,
        max_position_embeddings=CONFIG_MAX_POSITION_EMBEDDINGS,
        vocab_size=CONFIG_VOCAB_SIZE,
    )

    # Create fixed input for initialization
    input_ids = create_fixed_input(CONFIG_BATCH_SIZE, CONFIG_SEQ_LEN, CONFIG_VOCAB_SIZE)
    print(f"Input shape for init: {input_ids.shape}")

    # Initialize parameters with fixed seed on CPU
    key = jax.random.PRNGKey(42)
    params = model.init(key, input_ids, deterministic=False)
    
    param_count = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"Model parameters: {param_count / 1e6:.2f}M")

    # Save parameters
    save_filepath = get_model_params_path()
    save_params_with_npz(params, save_filepath)
    print(f"\nParameters saved to: {save_filepath}")


if __name__ == "__main__":
    main()
