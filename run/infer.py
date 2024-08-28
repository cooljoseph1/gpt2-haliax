import haliax as hax
from haliax import NamedArray
import jax
import equinox as eqx

from typing import Optional, Iterator

from  gpt2 import Gpt2State
from .load_gpt2 import gpt2_model, VocabAxis
from .tokenizer import gpt2_tokenizer

@jax.jit
def _get_next_id_and_state(
        id_sequence: NamedArray,
        key: jax.random.PRNGKey,
        state: Optional[Gpt2State] = None
) -> tuple[NamedArray, Gpt2State]:
    """Given a sequence of ids (encoded tokens), generate a new id and return the new state"""

    # logits for all the tokens
    all_logits, state = gpt2_model(
        id_sequence,
        PositionAxis="position",
        inference=True,
        state=state,
        return_state=True
    )
    # logits for the new token only
    logits = all_logits["position", -1:]

    new_id = hax.random.categorical(
        key=key,
        logits=logits,
        axis=VocabAxis
    )
    return new_id, state

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
        next_id, state = _get_next_id_and_state(id_sequence=id_sequence, key=subkey, state=state)
        state = Gpt2State.align_to_chunks(state)

        yield next_id.item()
        id_sequence = next_id
        i += 1


def infer_bytes(text: str, *, key: jax.random.PRNGKey, num_tokens: int = -1):
    """Infer the next `num_tokens` tokens for the given text, returning the result as a string"""
    ids = gpt2_tokenizer.encode(text)
    id_sequence = hax.named(ids, "position")
    for id in generate_ids(id_sequence, key=key, num_ids=num_tokens):
        yield gpt2_tokenizer.decode_single_token_bytes(id)
