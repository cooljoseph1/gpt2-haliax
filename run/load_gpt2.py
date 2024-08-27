import pytree2safetensors

from .gpt2_skeleton import gpt2_skeleton, Gpt2, VocabAxis # Alias VocabAxis here

gpt2_model: Gpt2 = pytree2safetensors.load_into_pytree(
    gpt2_skeleton,
    "weights/gpt2.safetensors"
)