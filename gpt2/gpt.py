import jax
import equinox as eqx
import haliax as hax
from haliax.jax_utils import shaped_rng_split
from haliax import Axis, AxisSelector, NamedArray

from typing import NamedTuple, Optional

from .transformer import Transformer, TransformerConfig
from .embedder import Embedder, EmbedderConfig
from .layer_norm import LayerNorm, LayerNormConfig


class Gpt2Config(NamedTuple):
    embedder_config: EmbedderConfig
    num_layers: int # Number of transformer layers
    transformer_config: TransformerConfig
    final_layer_norm_config: LayerNormConfig
    dropout_prob: float = 0.1


class Gpt2(eqx.Module):
    """The entire GPT-2 model"""

    embedder: Embedder
    embed_dropout: hax.nn.Dropout
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

        embed_dropout = hax.nn.Dropout(pdrop=config.dropout_prob)

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
            embed_dropout=embed_dropout,
            transformers=transformers,
            final_layer_norm=final_layer_norm
        )
    
    @jax.named_call
    def __call__(
        self,
        input_ids: NamedArray,
        *,
        PositionAxis: AxisSelector,
        inference: bool = True,
        key: Optional[jax.random.PRNGKey] = None
    ) -> NamedArray:
        
        if not inference:
            assert key is not None, "A PRNGKey must be provided during training for dropout"

        k_dropout, key = jax.random.split(key, 2) if key is not None else (None, None)

        # embed
        embeddings = self.embedder.embed(input_ids, PositionAxis=PositionAxis)
        # dropout on the initial embeddings
        embeddings = self.embed_dropout(embeddings, inference=inference, key=k_dropout)


        for transformer in self.transformers:
            key, subkey = jax.random.split(key, 2) if key is not None else (None, None)
            embeddings = transformer(
                embeddings,
                PositionAxis=PositionAxis,
                inference=inference,
                key=subkey
            )

        normed_embeddings = self.final_layer_norm(embeddings)
        logits = self.embedder.unembed(normed_embeddings)
        return logits