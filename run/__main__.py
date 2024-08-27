import sys
import jax
from .infer import infer_bytes

for token_bytes in infer_bytes(
    "The sun is the center of the solar system.",
    num_tokens=-1,
    key=jax.random.key(5)
):
    sys.stdout.buffer.write(token_bytes)
    sys.stdout.flush()