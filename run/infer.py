import haliax as hax
from haliax import NamedArray
import jax

from typing import Optional, Iterator

from  gpt2 import Gpt2State
from .load_gpt2 import gpt2_model, VocabAxis
from .tokenizer import gpt2_tokenizer


def get_logits(
    id_sequence: NamedArray,
    state: Optional[Gpt2State] = None
) -> NamedArray:
    """Given a sequence of ids (encoded tokens), output the logits for the next id"""
    # logits for all the tokens
    all_output_logits, state = gpt2_model(
        id_sequence,
        PositionAxis="position",
        inference=True,
        state=state,
        return_state=True
    )
    output_logits = all_output_logits["position", -1:] # logits for the new token
    return output_logits, state

def sample_id(id_logits: NamedArray, key: jax.random.PRNGKey):
    choice = hax.random.categorical(
        key=key,
        logits=id_logits,
        axis=VocabAxis
    )
    return choice

def generate_ids(
    id_sequence: NamedArray,
    *,
    key: jax.random.PRNGKey,
    num_ids: int = -1, # How many new tokens should you generate? Defaults to -1, which means it will generate forever
    state: Optional[Gpt2State] = None
) -> Iterator[int]:
     i = 0
     while i != num_ids:
        key, subkey = jax.random.split(key, 2)
        next_logits, state = get_logits(id_sequence, state=state)
        next_id = sample_id(id_logits=next_logits, key=subkey)

        yield next_id.item()
        id_sequence = next_id
        i += 1

def infer_bytes(text: str, *, key: jax.random.PRNGKey, num_tokens: int = -1):
    """Infer the next `num_tokens` tokens for the given text, returning the result as a string"""
    ids = gpt2_tokenizer.encode(text)
    id_sequence = hax.named(ids, "position")
    for id in generate_ids(id_sequence, key=key, num_ids=num_tokens):
        yield gpt2_tokenizer.decode_single_token_bytes(id)
