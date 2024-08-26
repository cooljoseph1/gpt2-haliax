import jax
import equinox as eqx

import haliax as hax
from haliax import Axis, AxisSelector, AxisSpec, NamedArray

from typing import ClassVar, NamedTuple

class AttentionConfig(NamedTuple):
    num_heads: int # Number of heads in the multi-head attention
    head_embed_size: int # Dimension of each of the queries/keys/values in each head
    use_bias: bool = True # Use bias in linear layers?


class AttentionBlock(eqx.Module):
    """A single transformer causal attention block."""

    QKVAxis: ClassVar[AxisSelector] = Axis("qkv", 3)
    
    HeadsAxis: Axis = eqx.field(static=True) # Axis along which the heads are spread
    QueryKeyEmbedAxis: Axis = eqx.field(static=True) # Axis of the query/key along which dot products occur
    ValueEmbedAxis: Axis = eqx.field(static=True)

    project_qkv: hax.nn.Linear # Given input embeddings, generate the queries, keys, and values
    project_out: hax.nn.Linear # Combine the values after attention and project back out to the EmbedAxis

    @staticmethod
    def init(
        EmbedAxis: AxisSpec,
        *,
        config: AttentionConfig,
        key: jax.random.PRNGKey,
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
            project_out=project_out
        )

    @jax.named_call
    def __call__(
        self,
        input_sequence: NamedArray,
        *,
        PositionAxis: AxisSelector
    ) -> NamedArray:

        qkv = self.project_qkv(input_sequence)
        queries, keys, values = qkv.unbind(AttentionBlock.QKVAxis)
        attended = hax.nn.attention.self_attention(
            Pos=PositionAxis,
            Key=self.QueryKeyEmbedAxis,
            query=queries,
            key=keys,
            value=values,
            is_causal=True
        )
        out = self.project_out(attended)
        return out