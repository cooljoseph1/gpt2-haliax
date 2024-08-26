import jax
import equinox as eqx

from haliax import AxisSelector, AxisSpec, NamedArray

from typing import NamedTuple

from .attention import AttentionBlock, AttentionConfig
from .feed_forward import FeedForwardBlock, FeedForwardConfig
from ..layer_norm import LayerNorm, LayerNormConfig

class TransformerConfig(NamedTuple):
    layer_norm1_config: LayerNormConfig
    attention_config: AttentionConfig
    layer_norm2_config: LayerNormConfig
    feed_forward_config: FeedForwardConfig

class Transformer(eqx.Module):
    """A single transformer layer."""

    layer_norm1: LayerNorm
    attention: AttentionBlock
    layer_norm2: LayerNorm
    feed_forward: FeedForwardBlock

    @staticmethod
    def init(
        EmbedAxis: AxisSpec,
        *,
        config: TransformerConfig,
        key: jax.random.PRNGKey,
    ) -> "Transformer":
        attention_key, feed_forward_key = jax.random.split(key)

        attention_block = AttentionBlock.init(
            EmbedAxis=EmbedAxis,
            config=config.attention_config,
            key=attention_key,
        )
        feed_forward_block = FeedForwardBlock.init(
            InAxis=EmbedAxis,
            OutAxis=EmbedAxis,
            config=config.feed_forward_config,
            key=feed_forward_key,
        )

        layer_norm1 = LayerNorm.init(
            axis=EmbedAxis,
            eps=config.layer_norm1_config.eps,
            use_weight=config.layer_norm1_config.use_weight,
            use_bias=config.layer_norm1_config.use_bias
        )

        layer_norm2 = LayerNorm.init(
            axis=EmbedAxis,
            eps=config.layer_norm2_config.eps,
            use_weight=config.layer_norm2_config.use_weight,
            use_bias=config.layer_norm2_config.use_bias
        )

        return Transformer(
            layer_norm1=layer_norm1,
            attention=attention_block,
            layer_norm2=layer_norm2,
            feed_forward=feed_forward_block
        )

    @jax.named_call
    def __call__(
        self,
        input_sequence: NamedArray,
        *,
        PositionAxis: AxisSelector
    ) -> NamedArray:
        normed_input = self.layer_norm1(input_sequence)
        attention_output = self.attention(normed_input, PositionAxis=PositionAxis)
        attention_output = input_sequence + attention_output

        normed_attention = self.layer_norm2(attention_output)
        output_sequence = self.feed_forward(normed_attention)
        output_sequence = attention_output + output_sequence

        return output_sequence
