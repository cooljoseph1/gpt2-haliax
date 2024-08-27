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
        key: Optional[jax.random.PRNGKey] = None
    ) -> NamedArray:

        qkv = self.project_qkv(input_sequence)
        queries: NamedArray
        queries, keys, values = qkv.unbind(AttentionBlock.QKVAxis)

        # Change the queries' position axis' name to a different name. This is to avoid accidentally
        # doing a dot product along the position axes.
        PositionAxis = input_sequence.resolve_axis(PositionAxis) # Resolve the PositionAxis to an Axis (because it might have been a str)
        key_value_pos_axis_name = "key_value_" + PositionAxis.name
        keys = keys.rename({PositionAxis: key_value_pos_axis_name})
        values = values.rename({PositionAxis: key_value_pos_axis_name})
        KeyPositionAxis = keys.resolve_axis(key_value_pos_axis_name)

        # Make the causal mask
        causal_mask = hax.arange(PositionAxis).broadcast_axis(KeyPositionAxis) >= hax.arange(KeyPositionAxis)

        # How similar is each query to each key?
        scores = hax.dot(queries, keys, axis=self.QueryKeyEmbedAxis) / jax.numpy.sqrt(self.QueryKeyEmbedAxis.size)
        # Apply causal mask
        scores = scores - 1.0E9 * (1.0 - causal_mask)
        # Convert to weights (weighted so that the sum along the key's position axis is 1.0)
        weights = hax.nn.softmax(scores, axis=KeyPositionAxis)

        # Apply dropout
        weights = self.attention_weights_dropout(weights, inference=inference, key=key)

        # Sum the values with the weights
        answers = hax.dot(weights, values, axis=KeyPositionAxis)
    
        out = self.project_out(answers)
        return out