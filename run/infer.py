import haliax as hax
from haliax import NamedArray, Axis
import jax

from .load_gpt2 import gpt2_model, VocabAxis
from .tokenizer import gpt2_tokenizer


def get_logits(id_sequence: NamedArray):
    """Given a sequence of ids (encoded tokens), output the logits for the next id"""
    all_output_logits = gpt2_model(id_sequence, PositionAxis="position")  # logits for all the tokens
    output_logits = all_output_logits[{"position": -1}] # logits for the new token
    return output_logits

def sample_id(id_logits: NamedArray, key: jax.random.PRNGKey):
    choice = hax.random.categorical(
        key=key,
        logits=id_logits,
        axis=VocabAxis
    )
    return choice

def generate_ids(id_sequence: NamedArray, num_ids: int, key: jax.random.PRNGKey):
    for _ in range(num_ids):
        key, subkey = jax.random.split(key, 2)
        next_logits = get_logits(id_sequence)
        next_id = sample_id(id_logits=next_logits, key=subkey)

        yield next_id.item()

        id_sequence = hax.concatenate(
            "position",
            [
                id_sequence,
                next_id.broadcast_axis(Axis("position", 1))
            ]
        )

def infer_bytes(text: str, *, num_tokens: int, key: jax.random.PRNGKey):
    """Infer the next `num_tokens` tokens for the given text, returning the result as a string"""
    ids = gpt2_tokenizer.encode(text)
    id_sequence = hax.named(ids, "position")
    for id in generate_ids(id_sequence, num_tokens, key):
        yield gpt2_tokenizer.decode_single_token_bytes(id)


if __name__ == "__main__":
    import sys

    for token_bytes in infer_bytes(
        "The sun is the center of the solar system.",
        num_tokens=100,
        key=jax.random.key(5)
    ):
        sys.stdout.buffer.write(token_bytes)
        sys.stdout.flush()