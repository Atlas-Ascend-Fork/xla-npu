import os
import traceback

import jax
import jax.numpy as jnp
import numpy as np

# Note: No need for ctypes.CDLL() anymore, JAX loads the library automatically with the correct ENV var
# 1. Define the platform name and library path
PLATFORM_NAME = 'npu'  # Use a simple name without underscores for safety
LIBRARY_PATH = os.path.abspath("pjrt_c_api_npu_plugin.so")

# 2. Set the mandatory environment variable
#    The format is 'platform_name:path/to/library.so'
os.environ['PJRT_NAMES_AND_LIBRARY_PATHS'] = f"{PLATFORM_NAME}:{LIBRARY_PATH}"

# 3. (Optional but good practice) Ensure JAX prefers your platform
#    This is the variable you used before, useful to explicitly request it
os.environ['JAX_PLATFORMS'] = PLATFORM_NAME

def test_add():
    print("=" * 50)
    print("Testing Add Operation")
    print("=" * 50)
    
    x1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    x2 = np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
    
    print(f"Input x1: {x1}")
    print(f"Input x2: {x2}")
    
    def add_func(x1, x2):
        return x1 + x2
    
    func = jax.jit(add_func)
    result = func(x1, x2)
    print(f"Output: {result}")
    print()


def test_tanh():
    print("=" * 50)
    print("Testing Tanh Operation")
    print("=" * 50)
    
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    
    print(f"Input x: {x}")
    
    def tanh_func(x):
        return jnp.tanh(x)
    
    func = jax.jit(tanh_func)
    result = func(x)
    print(f"Output: {result}")
    print()


def test_multiply():
    print("=" * 50)
    print("Testing Multiply Operation")
    print("=" * 50)
    
    x1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    x2 = np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
    
    print(f"Input x1: {x1}")
    print(f"Input x2: {x2}")
    
    def multiply_func(x1, x2):
        return x1 * x2
    
    func = jax.jit(multiply_func)
    result = func(x1, x2)
    print(f"Output: {result}")
    print()


def test_dot():
    print("=" * 50)
    print("Testing Dot Operation")
    print("=" * 50)
    
    x1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    x2 = np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    
    print(f"Input x1: {x1}")
    print(f"Input x2: {x2}")
    
    def dot_func(x1, x2):
        return jnp.dot(x1, x2)
    
    func = jax.jit(dot_func)
    result = func(x1, x2)
    print(f"Output: {result}")
    print()


def test_divide():
    print("=" * 50)
    print("Testing Divide Operation")
    print("=" * 50)
    
    x1 = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    x2 = np.array([[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]])
    
    print(f"Input x1: {x1}")
    print(f"Input x2: {x2}")
    
    def divide_func(x1, x2):
        return x1 / x2
    
    func = jax.jit(divide_func)
    result = func(x1, x2)
    print(f"Output: {result}")
    print()


def test_subtract():
    print("=" * 50)
    print("Testing Subtract Operation")
    print("=" * 50)
    
    x1 = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    x2 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    
    print(f"Input x1: {x1}")
    print(f"Input x2: {x2}")
    
    def subtract_func(x1, x2):
        return x1 - x2
    
    func = jax.jit(subtract_func)
    result = func(x1, x2)
    print(f"Output: {result}")
    print()


def test_negate():
    print("=" * 50)
    print("Testing Negate Operation")
    print("=" * 50)
    
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    
    print(f"Input x: {x}")
    
    def negate_func(x):
        return -x
    
    func = jax.jit(negate_func)
    result = func(x)
    print(f"Output: {result}")
    print()


def test_exp():
    print("=" * 50)
    print("Testing Exp Operation")
    print("=" * 50)
    
    x = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    
    print(f"Input x: {x}")
    
    def exp_func(x):
        return jnp.exp(x)
    
    func = jax.jit(exp_func)
    result = func(x)
    print(f"Output: {result}")
    print()


def test_maximum():
    print("=" * 50)
    print("Testing Maximum Operation")
    print("=" * 50)
    
    x1 = np.array([[1.0, 5.0, 3.0], [7.0, 2.0, 9.0]])
    x2 = np.array([[4.0, 2.0, 6.0], [3.0, 8.0, 1.0]])
    
    print(f"Input x1: {x1}")
    print(f"Input x2: {x2}")
    
    def maximum_func(x1, x2):
        return jnp.maximum(x1, x2)
    
    func = jax.jit(maximum_func)
    result = func(x1, x2)
    print(f"Output: {result}")
    print()


def test_concatenate():
    print("=" * 50)
    print("Testing Concatenate Operation")
    print("=" * 50)
    
    x1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    x2 = np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
    
    print(f"Input x1: {x1}")
    print(f"Input x2: {x2}")
    
    def concatenate_func(x1, x2):
        return jnp.concatenate([x1, x2], axis=0)
    
    func = jax.jit(concatenate_func)
    result = func(x1, x2)
    print(f"Output: {result}")
    print()


def test_reduce_max():
    print("=" * 50)
    print("Testing ReduceMax Operation")
    print("=" * 50)
    
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    
    print(f"Input x: {x}")
    
    def reduce_max_func(x):
        return jnp.max(x, axis=0)
    
    func = jax.jit(reduce_max_func)
    result = func(x)
    print(f"Output: {result}")
    print()


def test_reduce_sum():
    print("=" * 50)
    print("Testing ReduceSum Operation")
    print("=" * 50)
    
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    
    print(f"Input x: {x}")
    
    def reduce_sum_func(x):
        return jnp.sum(x, axis=0)
    
    func = jax.jit(reduce_sum_func)
    result = func(x)
    print(f"Output: {result}")
    print()


if __name__ == "__main__":
    print("Starting JAX Operations Tests")
    print("=" * 50)
    print()
    
    try:
        test_add()
        test_tanh()
        test_multiply()
        test_dot()
        test_divide()
        test_subtract()
        test_negate()
        test_exp()
        test_maximum()
        test_concatenate()
        test_reduce_max()
        test_reduce_sum()
        
        print("=" * 50)
        print("All tests completed successfully!")
        print("=" * 50)
    except Exception as e:
        print(f"Error occurred: {e}")
        traceback.print_exc()
