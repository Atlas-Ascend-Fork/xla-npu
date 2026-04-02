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
os.environ["JAX_PLATFORM"] = "cpu"

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Tuple
import numpy as np
from safetensors import safe_open
from transformers import AutoTokenizer

class Qwen2Config:
    def __init__(
        self,
        hidden_size=896,
        num_hidden_layers=2,
        num_attention_heads=14,
        num_key_value_heads=2,
        intermediate_size=4864,
        max_position_embeddings=32768,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        vocab_size=151936,
        hidden_act="silu",
        attention_dropout=0.0,
        **kwargs
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.vocab_size = vocab_size
        self.hidden_act = hidden_act
        self.attention_dropout = attention_dropout
        self.head_dim = hidden_size // num_attention_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads


class Qwen2RMSNorm(nn.Module):
    hidden_size: int
    eps: float = 1e-6

    def setup(self):
        self.weight = self.param('weight', nn.initializers.ones, (self.hidden_size,))

    def __call__(self, hidden_states):
        variance = jnp.mean(hidden_states ** 2, axis=-1, keepdims=True)
        hidden_states = hidden_states * jax.lax.rsqrt(variance + self.eps)
        return self.weight * hidden_states


class Qwen2RotaryEmbedding(nn.Module):
    config: Qwen2Config

    def setup(self):
        dim = self.config.head_dim
        base = self.config.rope_theta

        inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2) / dim))
        self.inv_freq = inv_freq

    def __call__(self, x, position_ids):
        batch_size, seq_len = position_ids.shape

        inv_freq = self.inv_freq
        position_ids_float = position_ids.astype(jnp.float32)

        freqs = jnp.outer(position_ids_float, inv_freq)
        freqs = freqs.reshape(batch_size, seq_len, -1)

        emb = jnp.concatenate((freqs, freqs), axis=-1)
        cos = jnp.cos(emb)
        sin = jnp.sin(emb)

        return cos.astype(x.dtype), sin.astype(x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: jnp.ndarray, n_rep: int) -> jnp.ndarray:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].repeat(n_rep, axis=2)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Qwen2Attention(nn.Module):
    config: Qwen2Config
    layer_idx: int

    def setup(self):
        self.head_dim = self.config.head_dim
        self.num_key_value_groups = self.config.num_key_value_groups
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Dense(
            self.config.num_attention_heads * self.head_dim,
            use_bias=True
        )
        self.k_proj = nn.Dense(
            self.config.num_key_value_heads * self.head_dim,
            use_bias=True
        )
        self.v_proj = nn.Dense(
            self.config.num_key_value_heads * self.head_dim,
            use_bias=True
        )
        self.o_proj = nn.Dense(
            self.config.num_attention_heads * self.head_dim,
            use_bias=False
        )

    def __call__(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        deterministic=True
    ):
        batch_size, seq_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.reshape(batch_size, seq_len, self.config.num_attention_heads, self.head_dim).transpose(0, 2, 1, 3)
        key_states = key_states.reshape(batch_size, seq_len, self.config.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)
        value_states = value_states.reshape(batch_size, seq_len, self.config.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = jnp.matmul(query_states, key_states.transpose(0, 1, 3, 2)) * self.scaling

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        attn_output = jnp.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(0, 2, 1, 3)

        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output


class Qwen2MLP(nn.Module):
    config: Qwen2Config

    def setup(self):
        self.gate_proj = nn.Dense(self.config.intermediate_size, use_bias=False)
        self.up_proj = nn.Dense(self.config.intermediate_size, use_bias=False)
        self.down_proj = nn.Dense(self.config.hidden_size, use_bias=False)

    def __call__(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(jax.nn.silu(gate) * up)


class Qwen2DecoderLayer(nn.Module):
    config: Qwen2Config
    layer_idx: int

    def setup(self):
        self.self_attn = Qwen2Attention(config=self.config, layer_idx=self.layer_idx)
        self.mlp = Qwen2MLP(config=self.config)
        self.input_layernorm = Qwen2RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        position_embeddings=None,
        deterministic=True
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            deterministic=deterministic
        )

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen2Model(nn.Module):
    config: Qwen2Config

    def setup(self):
        self.embed_tokens = nn.Embed(self.config.vocab_size, self.config.hidden_size)
        self.layers = [
            Qwen2DecoderLayer(self.config, layer_idx=i)
            for i in range(self.config.num_hidden_layers)
        ]
        self.norm = Qwen2RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(self.config)

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        deterministic=True
    ):
        batch_size, seq_len = input_ids.shape

        if position_ids is None:
            position_ids = jnp.arange(seq_len)[None, :]

        hidden_states = self.embed_tokens(input_ids)

        cos, sin = self.rotary_emb(hidden_states, position_ids)
        position_embeddings = (cos, sin)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                deterministic=deterministic
            )

        hidden_states = self.norm(hidden_states)

        return hidden_states


class Qwen2ForCausalLM(nn.Module):
    config: Qwen2Config

    def setup(self):
        self.model = Qwen2Model(self.config)
        self.lm_head = nn.Dense(self.config.vocab_size, use_bias=False)

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        deterministic=True
    ):
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        position_ids=position_ids,
            deterministic=deterministic
        )

        logits = self.lm_head(hidden_states)
        return logits


def create_causal_mask(seq_len):
    mask = jnp.triu(jnp.ones((seq_len, seq_len)), k=1)
    mask = mask * -1e10
    return mask[None, None, :, :]


def load_safetensors_weights(model_path):
    """加载safetensors权重文件"""
    weights = {}
    with safe_open(model_path, framework='pt', device='cpu') as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            weights[key] = tensor.float().numpy()
    return weights


def convert_pytorch_to_jax_weights(pytorch_weights, config, jax_params):
    """将PyTorch权重转换为JAX格式"""
    jax_weights = jax.tree_util.tree_map(lambda x: x, jax_params)

    for key, value in pytorch_weights.items():
        if key == 'model.embed_tokens.weight':
            jax_weights['params']['model']['embed_tokens']['embedding'] = value
        elif key == 'model.norm.weight':
            jax_weights['params']['model']['norm']['weight'] = value
        elif key == 'lm_head.weight':
            jax_weights['params']['lm_head']['kernel'] = value.T
        elif key.startswith('model.layers.'):
            parts = key.split('.')
            layer_idx = int(parts[2])
            layer_key = f'layers_{layer_idx}'

            if layer_idx >= config.num_hidden_layers:
                continue

            if parts[3] == 'input_layernorm' and parts[4] == 'weight':
                jax_weights['params']['model'][layer_key]['input_layernorm']['weight'] = value
            elif parts[3] == 'post_attention_layernorm' and parts[4] == 'weight':
                jax_weights['params']['model'][layer_key]['post_attention_layernorm']['weight'] = value
            elif parts[3] == 'mlp':
                if parts[4] == 'gate_proj' and parts[5] == 'weight':
                    jax_weights['params']['model'][layer_key]['mlp']['gate_proj']['kernel'] = value.T
                elif parts[4] == 'up_proj' and parts[5] == 'weight':
                    jax_weights['params']['model'][layer_key]['mlp']['up_proj']['kernel'] = value.T
                elif parts[4] == 'down_proj' and parts[5] == 'weight':
                    jax_weights['params']['model'][layer_key]['mlp']['down_proj']['kernel'] = value.T
            elif parts[3] == 'self_attn':
                if parts[4] == 'q_proj':
                    if parts[5] == 'weight':
                        jax_weights['params']['model'][layer_key]['self_attn']['q_proj']['kernel'] = value.T
                    elif parts[5] == 'bias':
                        jax_weights['params']['model'][layer_key]['self_attn']['q_proj']['bias'] = value
                elif parts[4] == 'k_proj':
                    if parts[5] == 'weight':
                        jax_weights['params']['model'][layer_key]['self_attn']['k_proj']['kernel'] = value.T
                    elif parts[5] == 'bias':
                        jax_weights['params']['model'][layer_key]['self_attn']['k_proj']['bias'] = value
                elif parts[4] == 'v_proj':
                    if parts[5] == 'weight':
                        jax_weights['params']['model'][layer_key]['self_attn']['v_proj']['kernel'] = value.T
                    elif parts[5] == 'bias':
                        jax_weights['params']['model'][layer_key]['self_attn']['v_proj']['bias'] = value
                elif parts[4] == 'o_proj' and parts[5] == 'weight':
                    jax_weights['params']['model'][layer_key]['self_attn']['o_proj']['kernel'] = value.T

    return jax_weights


def sample_token(logits, temperature=1.0, top_k=50):
    """从logits中采样下一个token"""
    if temperature > 0:
        logits = logits / temperature

    if top_k > 0:
        top_k_logits, top_k_indices = jax.lax.top_k(logits, top_k)
        probs = jax.nn.softmax(top_k_logits, axis=-1)
        next_token = jax.random.categorical(jax.random.PRNGKey(0), probs)
        next_token = top_k_indices[next_token]
    else:
        probs = jax.nn.softmax(logits, axis=-1)
        next_token = jax.random.categorical(jax.random.PRNGKey(0), probs)

    return next_token


def generate_text(model, params, tokenizer, prompt, max_length=10, temperature=0.7, top_k=50):
    """生成文本"""
    print(f"输入: {prompt}")

    inputs = tokenizer(prompt, return_tensors="np")
    input_ids = jnp.array(inputs['input_ids'])

    generated_tokens = list(input_ids[0])

    for i in range(max_length):
        current_input_ids = jnp.array([generated_tokens])
        seq_len = current_input_ids.shape[1]

        attention_mask = create_causal_mask(seq_len)

        logits = model.apply(params, current_input_ids, attention_mask=attention_mask, deterministic=False)
        next_token_logits = logits[0, -1, :]

        next_token = sample_token(next_token_logits, temperature=temperature, top_k=top_k)
        next_token = int(next_token)

        generated_tokens.append(next_token)

        if next_token == tokenizer.eos_token_id:
            break

        print(f"生成进度: {i+1}/{max_length}")

    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return generated_text


def main():
    print("=== Qwen2 JAX端到端推理 ===")

    model_path = "/huawei/code/test/qwen_jax/model_cache/Qwen2.5-0.5B-Instruct"
    weights_path = f"{model_path}/model.safetensors"

    print("加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True
    )
    print("✓ tokenizer加载完成")

    print(f"加载权重文件: {weights_path}")
    pytorch_weights = load_safetensors_weights(weights_path)
    print(f"✓ 加载了 {len(pytorch_weights)} 个参数")

    config = Qwen2Config(
        hidden_size=896,
        num_hidden_layers=24,
        num_attention_heads=14,
        num_key_value_heads=2,
        intermediate_size=4864,
        max_position_embeddings=32768,
        vocab_size=151936
    )

    jax_model = Qwen2ForCausalLM(config)

    key = jax.random.PRNGKey(0)
    batch_size = 1
    seq_len = 8
    input_ids = jax.random.randint(key, (batch_size, seq_len), 0, config.vocab_size)

    print("初始化JAX模型参数...")
    params = jax_model.init(key, input_ids, deterministic=False)
    print("✓ JAX模型参数初始化完成")

    print("转换PyTorch权重到JAX格式...")
    jax_weights = convert_pytorch_to_jax_weights(pytorch_weights, config, params)
    print("✓ 权重转换完成")

    print("\n=== 开始文本生成 ===")
    prompt = "你好，中国的首都是哪里"
    generated_text = generate_text(
        jax_model,
        jax_weights,
        tokenizer,
        prompt,
        max_length=10,
        temperature=0.7,
        top_k=50
    )

    print(f"\n生成结果:")
    print(generated_text)

    print("\n✓ Qwen2 JAX端到端推理完成!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()