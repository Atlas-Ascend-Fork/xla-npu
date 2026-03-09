# -*- coding:utf-8 -*-
"""
JAX implementation of MMOE (Multi-gate Mixture-of-Experts)

Author: JAX Conversion
Based on PyTorch implementation by zanshuxun

Reference:
    [1] Jiaqi Ma, Zhe Zhao, Xinyang Yi, et al. Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts[C]
    https://dl.acm.org/doi/10.1145/3219819.3220007
"""

import os
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, List


class DNN(nn.Module):
    """Multi-Layer Perceptron.

    Args:
        hidden_units: List of positive integers, layer units in each layer.
        activation: Activation function to use.
        dropout_rate: Dropout rate.
        use_bn: Whether to use BatchNorm.
        use_bias: Whether to use bias in linear layers.
    """

    hidden_units: List[int]
    activation: str = 'relu'
    dropout_rate: float = 0.0
    use_bn: bool = False
    use_bias: bool = True
    eval_mode: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        for i in range(len(self.hidden_units) - 1):
            x = nn.Dense(
                features=self.hidden_units[i + 1],
                use_bias=self.use_bias,
                kernel_init=nn.initializers.normal(0.0001)
            )(x)

            if self.use_bn:
                x = nn.BatchNorm(momentum=0.1, epsilon=1e-5)(x, use_running_average=not train)

            if self.activation.lower() == 'sigmoid':
                x = jax.nn.sigmoid(x)
            elif self.activation.lower() == 'linear':
                pass
            elif self.activation.lower() == 'relu':
                x = nn.relu(x)
            elif self.activation.lower() == 'leaky_relu':
                x = jax.nn.leaky_relu(x, negative_slope=0.01)
            elif self.activation.lower() == 'tanh':
                x = jnp.tanh(x)
            elif self.activation.lower() == 'elu':
                x = jax.nn.elu(x)
            elif self.activation.lower() == 'selu':
                x = jax.nn.selu(x)
            elif self.activation.lower() == 'gelu':
                x = jax.nn.gelu(x)

            if self.dropout_rate > 0.0 and train:
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

        return x


class PredictionLayer(nn.Module):
    """Prediction layer for binary classification or regression tasks.

    Args:
        task: Task type - 'binary' or 'regression'.
        use_bias: Whether to add bias.
    """

    task: str = 'binary'
    use_bias: bool = True

    def setup(self):
        if self.use_bias:
            self.bias = self.param('bias', nn.initializers.zeros, (1,))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.use_bias:
            x = x + self.bias

        return jax.nn.sigmoid(x) if self.task == 'binary' else x


class MMOE(nn.Module):
    """Multi-gate Mixture-of-Experts architecture in JAX.

    Args:
        num_experts: Number of expert networks.
        num_tasks: Number of tasks.
        input_dim: Input feature dimension.
        expert_dnn_hidden_units: Hidden units for expert DNNs.
        gate_dnn_hidden_units: Hidden units for gate DNNs.
        tower_dnn_hidden_units: Hidden units for tower DNNs.
        dnn_activation: Activation function for DNNs.
        dnn_dropout: Dropout rate.
        dnn_use_bn: Whether to use BatchNorm.
        task_types: List of task types ['binary', 'regression', ...].
        task_names: List of task names.

    Returns:
        Output for all tasks concatenated.
    """

    num_experts: int = 3
    num_tasks: int = 2
    input_dim: int = 100
    expert_dnn_hidden_units: Tuple[int, ...] = (256, 128)
    gate_dnn_hidden_units: Tuple[int, ...] = (64,)
    tower_dnn_hidden_units: Tuple[int, ...] = (64,)
    dnn_activation: str = 'relu'
    dnn_dropout: float = 0.0
    dnn_use_bn: bool = False
    task_types: Tuple[str, ...] = ('binary', 'binary')
    task_names: Tuple[str, ...] = ('ctr', 'ctcvr')

    def setup(self):
        if self.num_tasks <= 1:
            raise ValueError("num_tasks must be greater than 1")
        if self.num_experts <= 1:
            raise ValueError("num_experts must be greater than 1")
        if len(self.task_types) != self.num_tasks:
            raise ValueError("num_tasks must be equal to the length of task_types")

        for task_type in self.task_types:
            if task_type not in ['binary', 'regression']:
                raise ValueError(f"task must be binary or regression, {task_type} is illegal")

        expert_input_dim = self.input_dim
        expert_hidden = [expert_input_dim] + list(self.expert_dnn_hidden_units)

        self.expert_dnns = [
            DNN(
                hidden_units=self.expert_dnn_hidden_units,
                activation=self.dnn_activation,
                dropout_rate=self.dnn_dropout,
                use_bn=self.dnn_use_bn
            )
            for _ in range(self.num_experts)
        ]

        gate_input_dim = self.input_dim
        if len(self.gate_dnn_hidden_units) > 0:
            self.gate_dnns = [
                DNN(
                    hidden_units=self.gate_dnn_hidden_units,
                    activation=self.dnn_activation,
                    dropout_rate=self.dnn_dropout,
                    use_bn=self.dnn_use_bn
                )
                for _ in range(self.num_tasks)
            ]
            gate_final_input_dim = self.gate_dnn_hidden_units[-1]
        else:
            gate_final_input_dim = gate_input_dim

        self.gate_final_layers = [
            nn.Dense(
                features=self.num_experts,
                use_bias=False,
                kernel_init=nn.initializers.normal(0.0001)
            )
            for _ in range(self.num_tasks)
        ]

        tower_input_dim = self.expert_dnn_hidden_units[-1]
        if len(self.tower_dnn_hidden_units) > 0:
            self.tower_dnns = [
                DNN(
                    hidden_units=self.tower_dnn_hidden_units,
                    activation=self.dnn_activation,
                    dropout_rate=self.dnn_dropout,
                    use_bn=self.dnn_use_bn
                )
                for _ in range(self.num_tasks)
            ]
            tower_final_input_dim = self.tower_dnn_hidden_units[-1]
        else:
            tower_final_input_dim = tower_input_dim

        self.tower_final_layers = [
            nn.Dense(
                features=1,
                use_bias=False,
                kernel_init=nn.initializers.normal(0.0001)
            )
            for _ in range(self.num_tasks)
        ]

        self.prediction_layers = [
            PredictionLayer(task=task_type, use_bias=True)
            for task_type in self.task_types
        ]

    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        batch_size = x.shape[0]

        expert_outputs = []
        for expert_dnn in self.expert_dnns:
            expert_out = expert_dnn(x, train=train)
            expert_outputs.append(expert_out)
        expert_outputs = jnp.stack(expert_outputs, axis=1)

        mmoe_outputs = []
        for task_idx in range(self.num_tasks):
            if len(self.gate_dnn_hidden_units) > 0:
                gate_dnn_out = self.gate_dnns[task_idx](x, train=train)
                gate_weights = self.gate_final_layers[task_idx](gate_dnn_out)
            else:
                gate_weights = self.gate_final_layers[task_idx](x)

            gate_weights = jax.nn.softmax(gate_weights, axis=-1)
            gate_weights = gate_weights[:, :, jnp.newaxis]

            mmoe_out = jnp.sum(gate_weights * expert_outputs, axis=1)
            mmoe_outputs.append(mmoe_out)

        task_outputs = []
        for task_idx in range(self.num_tasks):
            if len(self.tower_dnn_hidden_units) > 0:
                tower_out = self.tower_dnns[task_idx](mmoe_outputs[task_idx], train=train)
                tower_logit = self.tower_final_layers[task_idx](tower_out)
            else:
                tower_logit = self.tower_final_layers[task_idx](mmoe_outputs[task_idx])

            prediction = self.prediction_layers[task_idx](tower_logit)
            task_outputs.append(prediction)

        output = jnp.concatenate(task_outputs, axis=-1)

        return output


def create_mmoe(
    input_dim: int,
    num_experts: int = 3,
    num_tasks: int = 2,
    expert_dnn_hidden_units: Tuple[int, ...] = (256, 128),
    gate_dnn_hidden_units: Tuple[int, ...] = (64,),
    tower_dnn_hidden_units: Tuple[int, ...] = (64,),
    dnn_activation: str = 'relu',
    dnn_dropout: float = 0.0,
    dnn_use_bn: bool = False,
    task_types: Tuple[str, ...] = ('binary', 'binary'),
    task_names: Tuple[str, ...] = ('ctr', 'ctcvr'),
) -> MMOE:
    """Create the MMOE model.

    Args:
        input_dim: Input feature dimension.
        num_experts: Number of expert networks.
        num_tasks: Number of tasks.
        expert_dnn_hidden_units: Hidden units for expert DNNs.
        gate_dnn_hidden_units: Hidden units for gate DNNs.
        tower_dnn_hidden_units: Hidden units for tower DNNs.
        dnn_activation: Activation function.
        dnn_dropout: Dropout rate.
        dnn_use_bn: Whether to use BatchNorm.
        task_types: Task types tuple.
        task_names: Task names tuple.

    Returns:
        MMOE model instance.
    """
    model = MMOE(
        num_experts=num_experts,
        num_tasks=num_tasks,
        input_dim=input_dim,
        expert_dnn_hidden_units=expert_dnn_hidden_units,
        gate_dnn_hidden_units=gate_dnn_hidden_units,
        tower_dnn_hidden_units=tower_dnn_hidden_units,
        dnn_activation=dnn_activation,
        dnn_dropout=dnn_dropout,
        dnn_use_bn=dnn_use_bn,
        task_types=task_types,
        task_names=task_names,
    )

    return model


def save_params_with_npz(params, filepath: str):
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

    print(f"参数已保存到: {filepath}")


def load_params_with_npz(filepath: str):
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


# Module-level JIT cache
_jit_cache = {}

def get_jitted_forward(model):
    """Get or create JIT-compiled forward function for the model."""
    # Use model ID as cache key
    model_id = id(model)
    if model_id not in _jit_cache:
        @jax.jit
        def jitted_forward(params, x):
            return model.apply(params, x, train=False)
        _jit_cache[model_id] = jitted_forward
    return _jit_cache[model_id]

def run_inference(model, params, x):
    """Run inference and return results."""
    # Clear JIT cache before each inference to force recompilation
    # This avoids stale cache issues and ensures fresh compilation each time
    _jit_cache.clear()
    
    jitted_forward = get_jitted_forward(model)
    output = jitted_forward(params, x)
    return np.array(output)


def print_results(output, device_name, batch_size, num_tasks):
    """打印推理结果."""
    print(f"\n{device_name} Output (first 5 samples):")
    print(output[:5])
    print(f"\n{device_name} Output statistics:")
    print(f"  Min: {output.min():.6f}")
    print(f"  Max: {output.max():.6f}")
    print(f"  Mean: {output.mean():.6f}")
    print(f"  Std: {output.std():.6f}")


# Model Configuration - Original small size
CONFIG_INPUT_DIM = 4
CONFIG_BATCH_SIZE = 2
CONFIG_NUM_EXPERTS = 2
CONFIG_NUM_TASKS = 2
CONFIG_EXPERT_DNN_HIDDEN_UNITS = (4, 2)
CONFIG_GATE_DNN_HIDDEN_UNITS = (2,)
CONFIG_TOWER_DNN_HIDDEN_UNITS = (2,)
CONFIG_DNN_ACTIVATION = 'relu'
CONFIG_TASK_TYPES = ('binary', 'binary')  # Must match CONFIG_NUM_TASKS
CONFIG_MODEL_PARAMS_FILE = "model_params.npz"

def get_model_params_path():
    """Get the path to model parameters file."""
    return os.path.join(os.path.dirname(__file__), CONFIG_MODEL_PARAMS_FILE)


def setup_model_and_data():
    """Setup model, load parameters and create input data."""
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

    # 使用随机数作为输入 (固定随机种子以保证可重复性)
    np.random.seed(42)
    x = np.random.rand(CONFIG_BATCH_SIZE, CONFIG_INPUT_DIM).astype(np.float32)

    return model, params, x


def run_inference_and_save(model, params, x, device_name, output_file):
    """Run inference and save results."""
    output = run_inference(model, params, x)

    np.savez(output_file, output=output, input=x)
    return output


def compare_results(output_cpu, output_npu, x_cpu, x_npu):
    """Compare CPU and NPU results."""
    if not np.allclose(x_cpu, x_npu):
        print("Warning: CPU and NPU inputs are different!")

    print("\n" + "=" * 60)
    print("--- Result Comparison (CPU vs NPU) ---")
    print("=" * 60)
    diff = np.abs(output_cpu - output_npu)
    rel_diff = diff / (np.abs(output_cpu) + 1e-8)
    print(f"Absolute difference (first 5 samples):")
    print(diff[:5])
    print(f"\nDifference statistics:")
    print(f"  Max absolute difference: {diff.max():.6e}")
    print(f"  Mean absolute difference: {diff.mean():.6e}")
    print(f"  Max relative difference: {rel_diff.max():.6e}")
    print(f"  Mean relative difference: {rel_diff.mean():.6e}")

    tolerance = 1e-4
    is_match = np.allclose(output_cpu, output_npu, rtol=tolerance, atol=tolerance)
    print(f"\nResults match (tolerance={tolerance}): {is_match}")
    return is_match
