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

"""
Initialize MMOE model parameters and save to file
Automatically reads configuration from model.py
"""

import os
import sys
import jax
import jax.numpy as jnp
import numpy as np

# Import configuration and utilities
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    CONFIG_INPUT_DIM,
    CONFIG_BATCH_SIZE,
    CONFIG_NUM_EXPERTS,
    CONFIG_NUM_TASKS,
    CONFIG_EXPERT_DNN_HIDDEN_UNITS,
    CONFIG_GATE_DNN_HIDDEN_UNITS,
    CONFIG_TOWER_DNN_HIDDEN_UNITS,
    CONFIG_DNN_ACTIVATION,
    CONFIG_TASK_TYPES,
    CONFIG_MODEL_PARAMS_FILE,
)
from model import create_mmoe


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
            - model: MMOE model instance
            - params: Model parameters
    """
    model = create_mmoe(
        input_dim=CONFIG_INPUT_DIM,
        num_experts=CONFIG_NUM_EXPERTS,
        num_tasks=CONFIG_NUM_TASKS,
        expert_dnn_hidden_units=CONFIG_EXPERT_DNN_HIDDEN_UNITS,
        gate_dnn_hidden_units=CONFIG_GATE_DNN_HIDDEN_UNITS,
        tower_dnn_hidden_units=CONFIG_TOWER_DNN_HIDDEN_UNITS,
        dnn_activation=CONFIG_DNN_ACTIVATION,
        task_types=CONFIG_TASK_TYPES,
    )
    
    save_filepath = get_model_params_path()
    params = load_params_with_npz(save_filepath)
    
    return model, params


def main():
    """Initialize MMOE model parameters and save to file."""
    print("Initializing MMOE model parameters...")
    print(f"Input dim: {CONFIG_INPUT_DIM}, Batch size: {CONFIG_BATCH_SIZE}")
    print(f"Num experts: {CONFIG_NUM_EXPERTS}, Num tasks: {CONFIG_NUM_TASKS}")
    print(f"Expert hidden units: {CONFIG_EXPERT_DNN_HIDDEN_UNITS}")
    print(f"Gate hidden units: {CONFIG_GATE_DNN_HIDDEN_UNITS}")
    print(f"Tower hidden units: {CONFIG_TOWER_DNN_HIDDEN_UNITS}")

    # Create model with configuration from model.py
    model = create_mmoe(
        input_dim=CONFIG_INPUT_DIM,
        num_experts=CONFIG_NUM_EXPERTS,
        num_tasks=CONFIG_NUM_TASKS,
        expert_dnn_hidden_units=CONFIG_EXPERT_DNN_HIDDEN_UNITS,
        gate_dnn_hidden_units=CONFIG_GATE_DNN_HIDDEN_UNITS,
        tower_dnn_hidden_units=CONFIG_TOWER_DNN_HIDDEN_UNITS,
        dnn_activation=CONFIG_DNN_ACTIVATION,
        task_types=CONFIG_TASK_TYPES,
    )

    # Initialize parameters with random seed
    key = jax.random.PRNGKey(42)
    dummy_input = jnp.ones((1, CONFIG_INPUT_DIM))
    key, init_key = jax.random.split(key)
    params = model.init(init_key, dummy_input, train=False)

    # Save parameters
    save_filepath = os.path.join(os.path.dirname(__file__), CONFIG_MODEL_PARAMS_FILE)
    save_params_with_npz(params, save_filepath)
    print(f"\nParameters saved to: {save_filepath}")


if __name__ == "__main__":
    main()
