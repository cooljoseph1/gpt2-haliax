import jax
import equinox as eqx
import haliax as hax
from haliax.jax_utils import shaped_rng_split
from haliax import Axis, AxisSelector, NamedArray

from typing import NamedTuple

from .transformer import Transformer, TransformerConfig
from .embedder import Embedder, EmbedderConfig
from .layer_norm import LayerNorm, LayerNormConfig


class Gpt2Config(NamedTuple):
    embedder_config: EmbedderConfig
    num_layers: int # Number of transformer layers
    transformer_config: TransformerConfig
    final_layer_norm_config: LayerNormConfig


class Gpt2(eqx.Module):
    """The entire GPT-2 model"""

    embedder: Embedder
    transformers: list[Transformer]
    final_layer_norm: LayerNorm

    @staticmethod
    def init(
        VocabAxis: Axis,
        *,
        config: Gpt2Config,
        key: jax.random.PRNGKey,
    ) -> "Gpt2":
        
        key_embedder, key_transformers = jax.random.split(key, 2)
        
        embedder = Embedder.init(
            VocabAxis=VocabAxis,
            config=config.embedder_config,
            key=key_embedder
        )

        EmbedAxis = embedder.EmbedAxis

        transformers = [
            Transformer.init(
                EmbedAxis=EmbedAxis,
                config=config.transformer_config,
                key=key
            )
            for key in jax.random.split(key_transformers, config.num_layers)
        ]

        final_layer_norm = LayerNorm.init(
            axis=EmbedAxis,
            eps=config.final_layer_norm_config.eps,
            use_weight=config.final_layer_norm_config.use_weight,
            use_bias=config.final_layer_norm_config.use_bias
        )

        return Gpt2(
            embedder=embedder,
            transformers=transformers,
            final_layer_norm=final_layer_norm
        )
    
    @jax.named_call
    def __call__(
        self,
        input_ids: NamedArray,
        *,
        PositionAxis: AxisSelector
    ) -> NamedArray:
        embeddings = self.embedder.embed(input_ids, PositionAxis=PositionAxis)
        for transformer in self.transformers:
            embeddings = transformer(embeddings, PositionAxis=PositionAxis)

        normed_embeddings = self.final_layer_norm(embeddings)
        logits = self.embedder.unembed(normed_embeddings)
        return logits