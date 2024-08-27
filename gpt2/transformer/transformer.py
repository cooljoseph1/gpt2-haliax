import jax
import equinox as eqx

import haliax as hax
from haliax import AxisSelector, AxisSpec, NamedArray

from typing import NamedTuple, Optional

from .attention import AttentionBlock, AttentionConfig, AttentionState
from .feed_forward import FeedForwardBlock, FeedForwardConfig
from ..layer_norm import LayerNorm, LayerNormConfig

class TransformerConfig(NamedTuple):
    layer_norm1_config: LayerNormConfig
    attention_config: AttentionConfig
    layer_norm2_config: LayerNormConfig
    feed_forward_config: FeedForwardConfig
    dropout_prob: float = 0.1


class TransformerState(eqx.Module):
    attention_state: AttentionState


class Transformer(eqx.Module):
    """A single transformer layer."""

    layer_norm1: LayerNorm
    attention: AttentionBlock
    attention_dropout: hax.nn.Dropout
    layer_norm2: LayerNorm
    feed_forward: FeedForwardBlock
    feed_forward_dropout: hax.nn.Dropout

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

        attention_dropout = hax.nn.Dropout(pdrop=config.dropout_prob)
        feed_forward_dropout = hax.nn.Dropout(pdrop=config.dropout_prob)

        return Transformer(
            layer_norm1=layer_norm1,
            attention=attention_block,
            attention_dropout=attention_dropout,
            layer_norm2=layer_norm2,
            feed_forward=feed_forward_block,
            feed_forward_dropout=feed_forward_dropout
        )

    @eqx.filter_jit
    @jax.named_call
    def __call__(
        self,
        input_sequence: NamedArray,
        *,
        PositionAxis: AxisSelector,
        inference: bool = True,
        key: Optional[jax.random.PRNGKey] = None,
        # To speed up inference, you can pass a state and ask for a state back
        state: Optional[TransformerState] = None,
        return_state: bool = False
    ) -> NamedArray:
        k_attention, k_dropout1, k_dropout2 = jax.random.split(key, 3) if key is not None else (None, None, None)

        # Attention block
        normed_input = self.layer_norm1(input_sequence)
        attention_output = self.attention(
            normed_input,
            PositionAxis=PositionAxis,
            inference=inference,
            key=k_attention,
            state=state.attention_state if state is not None else None,
            return_state=return_state
        )
        if return_state:
            attention_output, attention_state = attention_output
            new_state = TransformerState(attention_state=attention_state)

        # dropout
        attention_output = self.attention_dropout(
            attention_output,
            inference=inference,
            key=k_dropout1
        )
        attention_output = input_sequence + attention_output

        # Feed forward block
        normed_attention = self.layer_norm2(attention_output)
        output_sequence = self.feed_forward(normed_attention)
        # dropout
        output_sequence = self.feed_forward_dropout(
            output_sequence,
            inference=inference,
            key=k_dropout2
        )
        output_sequence = attention_output + output_sequence

        if return_state:
            return output_sequence, new_state
        else:
            return output_sequence
