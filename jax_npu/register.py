import os
import jax
import jax.numpy as jnp
import warnings
import traceback
import sys

# Note: No need for ctypes.CDLL() anymore, JAX loads the library automatically with the correct ENV var
# 1. Define the platform name and library path
PLATFORM_NAME = 'NPU' # Use a simple name without underscores for safety
LIBRARY_PATH = os.path.abspath("pjrt_c_api_npu_plugin.so")

# 2. Set the mandatory environment variable
#    The format is 'platform_name:path/to/library.so'
os.environ['PJRT_NAMES_AND_LIBRARY_PATHS'] = f"{PLATFORM_NAME}:{LIBRARY_PATH}"

# 3. (Optional but good practice) Ensure JAX prefers your platform
#    This is the variable you used before, useful to explicitly request it
os.environ['JAX_PLATFORMS'] = PLATFORM_NAME

# 4. Import jax.lib.xla_client to trigger the backend discovery mechanism
import jax.lib.xla_client
print(f"JAX Rerunning client discovery for platform: {PLATFORM_NAME}")


# --- Run a simple JAX program ---
try:
    # This should now print 'JAX Default Backend: mock_hw'
    # and trigger Mock_Client_Create() in your C++ code
    print(f"\nJAX Default Backend: {jax.default_backend()}")
    print(f"Devices type: {type(jax.devices())}")
    print(f"Devices available: {jax.devices()}")

    # Creates a JAX array, calling Mock_Buffer_FromHost()
    x = jnp.array([1.0, 2.0, 3.0])
    print(f"Created JAX array on device: {x.devices()}")

    # Compiles the operation using XLA, calling Mock_Compile()
    y = jax.jit(lambda x: x * 2)(x)

    # Triggers execution
    print(f"Computed result: {y.block_until_ready()}")

except RuntimeError as e:
    traceback.print_exc()
    print(f"\nJAX Integration Error: {e}")
    print("Check compilation steps and ensure all mandatory C++ functions are implemented.")