import jax
import equinox as eqx

import haliax as hax
from haliax import Axis, AxisSelector, AxisSpec, NamedArray

from typing import ClassVar, NamedTuple, Optional

class AttentionConfig(NamedTuple):
    num_heads: int # Number of heads in the multi-head attention
    head_embed_size: int # Dimension of each of the queries/keys/values in each head
    use_bias: bool = True # Use bias in linear layers?
    dropout_prob: float = 0.1


class AttentionState(eqx.Module):
    # Previously computed keys and values, possibily padded left with 0 vectors
    kv_cache: NamedArray

    PositionAxis: AxisSelector = eqx.field(static=True)
    # First index in kv_cache that contains actual data; the rest is zero vectors to pad to a consistent size
    first_index: int = 0
    
    # The size of a "chunk" to pad to
    chunk_size: ClassVar[int] = 1023
    KVAxis: ClassVar[AxisSelector] = Axis("kv", 2)

    def align_to_chunk_size(self) -> "AttentionState":
        kv = self.kv_cache
        PositionAxis = kv.resolve_axis(self.PositionAxis)
        length = PositionAxis.size
        used_length = length - self.first_index
        new_length = ((used_length - 1) // AttentionState.chunk_size + 1) * AttentionState.chunk_size
        NewPositionAxis = Axis(self.PositionAxis.name, new_length)

        if length > new_length:
            padded_kv = kv[PositionAxis, -new_length:]
        else:
            padded_kv = hax.pad_left(kv, PositionAxis, NewPositionAxis)
        
        new_first_index = NewPositionAxis.size - used_length

        return AttentionState(
            kv_cache=padded_kv,
            first_index=new_first_index,
            PositionAxis=NewPositionAxis
        )


class AttentionBlock(eqx.Module):
    """A single transformer causal attention block."""

    QKVAxis: ClassVar[AxisSelector] = Axis("qkv", 3)
    
    HeadsAxis: Axis = eqx.field(static=True) # Axis along which the heads are spread
    QueryKeyEmbedAxis: Axis = eqx.field(static=True) # Axis of the query/key along which dot products occur
    ValueEmbedAxis: Axis = eqx.field(static=True)

    project_qkv: hax.nn.Linear # Given input embeddings, generate the queries, keys, and values
    attention_weights_dropout: hax.nn.Dropout # Dropout during attention; drops out some of the weights in the weighted sum of values
    project_out: hax.nn.Linear # Combine the values after attention and project back out to the EmbedAxis

    @staticmethod
    def init(
        EmbedAxis: AxisSpec,
        *,
        config: AttentionConfig,
        key: jax.random.PRNGKey
    ):
        
        HeadsAxis = Axis("heads", config.num_heads)
        QueryKeyEmbedAxis = Axis("head_embed", config.head_embed_size)
        ValueEmbedAxis = QueryKeyEmbedAxis

        key_qkv, key_out = jax.random.split(key, 2)

        project_qkv = hax.nn.Linear.init(
            In=EmbedAxis,
            Out=(AttentionBlock.QKVAxis, HeadsAxis, QueryKeyEmbedAxis),
            key=key_qkv,
            use_bias=config.use_bias
        )

        attention_weights_dropout = hax.nn.Dropout(config.dropout_prob)

        project_out = hax.nn.Linear.init(
            In=(HeadsAxis, ValueEmbedAxis),
            Out=EmbedAxis,
            key=key_out,
            use_bias=config.use_bias
        )

        return AttentionBlock(
            HeadsAxis=HeadsAxis,
            QueryKeyEmbedAxis=QueryKeyEmbedAxis,
            ValueEmbedAxis=ValueEmbedAxis,
            project_qkv=project_qkv,
            attention_weights_dropout=attention_weights_dropout,
            project_out=project_out
        )

    @jax.named_call
    def __call__(
        self,
        input_sequence: NamedArray,
        *,
        PositionAxis: AxisSelector,
        inference: bool = True,
        key: Optional[jax.random.PRNGKey] = None,

        # To speed up inference, you can pass a state and ask for a state back
        state: Optional[AttentionState] = None,
        return_state: bool = False
    ) -> NamedArray:
        if not inference:
            assert key is not None, "When training, a PRNG key must be provided for dropout"

        # Generate the new queries, keys, and values
        qkv = self.project_qkv(input_sequence)
        queries, keys, values = qkv.unbind(AttentionBlock.QKVAxis)

        # Change the queries' position axis' name to a different name. This is to avoid accidentally
        # doing a dot product along the position axes.
        PositionAxis = input_sequence.resolve_axis(PositionAxis) # Resolve the PositionAxis to an Axis (because it might have been a str)
        key_value_pos_axis_name = "key_value_" + PositionAxis.name
        keys = keys.rename({PositionAxis: key_value_pos_axis_name})
        values = values.rename({PositionAxis: key_value_pos_axis_name})

        # Concatenate the previous values and keys
        if state is not None:
            previous_keys, previous_values = state.kv_cache[AttentionState.KVAxis, 0], state.kv_cache[AttentionState.KVAxis, 1]
            keys = hax.concatenate(key_value_pos_axis_name, [previous_keys, keys])
            values = hax.concatenate(key_value_pos_axis_name, [previous_values, values])

        # Resolve KeyValuePositionAxis to an Axis
        KeyValuePositionAxis = keys.resolve_axis(key_value_pos_axis_name)

        # Determine start_position for causal mask
        if state is None:
            start_position = 0
        else:
            start_position = state.kv_cache.resolve_axis(key_value_pos_axis_name).size

        # Make the causal mask
        causal_mask = hax.arange(PositionAxis, start=start_position).broadcast_axis(KeyValuePositionAxis) >= hax.arange(KeyValuePositionAxis)

        # How similar is each query to each key?
        scores = hax.dot(queries, keys, axis=self.QueryKeyEmbedAxis) / jax.numpy.sqrt(self.QueryKeyEmbedAxis.size)
        # Apply causal mask
        scores = scores - 1.0E9 * (1.0 - causal_mask)
        if state is not None:
            # Mask out the values in the cache that are just padding
            scores = scores - 1.0E9 * (hax.arange(KeyValuePositionAxis) < state.first_index)
        
        # Convert to weights (weighted so that the sum along the key's position axis is 1.0)
        weights = hax.nn.softmax(scores, axis=KeyValuePositionAxis)

        # Apply dropout
        weights = self.attention_weights_dropout(weights, inference=inference, key=key)

        # Sum the values with the weights
        answers = hax.dot(weights, values, axis=KeyValuePositionAxis)
    
        out = self.project_out(answers)

        if return_state:
            kv = hax.stack(AttentionState.KVAxis, [keys, values])
            new_state = AttentionState(
                kv_cache=kv,
                first_index=0 if state is None else state.first_index,
                PositionAxis=KeyValuePositionAxis
            )
            return out, new_state
        else:
            return out