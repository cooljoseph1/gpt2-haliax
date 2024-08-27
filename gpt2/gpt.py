import jax
import equinox as eqx
import haliax as hax
from haliax.jax_utils import shaped_rng_split
from haliax import Axis, AxisSelector, NamedArray

from typing import NamedTuple, Optional

from .transformer import Transformer, TransformerConfig, TransformerState
from .embedder import Embedder, EmbedderConfig
from .layer_norm import LayerNorm, LayerNormConfig


class Gpt2Config(NamedTuple):
    embedder_config: EmbedderConfig
    num_layers: int # Number of transformer layers
    transformer_config: TransformerConfig
    final_layer_norm_config: LayerNormConfig
    dropout_prob: float = 0.1


class Gpt2State(eqx.Module):
    transformer_states: list[TransformerState]
    first_position: int = eqx.field(static=True, default=0) # Position of the first new token to process


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
        key: Optional[jax.random.PRNGKey] = None,
        # To speed up inference, you can pass a state and ask for a state back
        state: Optional[Gpt2State] = None,
        return_state: bool = False
    ) -> NamedArray:
        
        if not inference:
            assert key is not None, "A PRNGKey must be provided during training for dropout"

        k_dropout, key = jax.random.split(key, 2) if key is not None else (None, None)

        first_position = 0 if state is None else state.first_position

        # embed
        embeddings = self.embedder.embed(input_ids, PositionAxis=PositionAxis, first_position=first_position)
        # dropout on the initial embeddings
        embeddings = self.embed_dropout(embeddings, inference=inference, key=k_dropout)

        if state is None:
            transformer_states = [None for _ in self.transformers]
        else:
            transformer_states = state.transformer_states
        new_transformer_states = []
        for transformer, transformer_state in zip(self.transformers, transformer_states):
            key, subkey = jax.random.split(key, 2) if key is not None else (None, None)
            embeddings = transformer(
                embeddings,
                PositionAxis=PositionAxis,
                inference=inference,
                key=subkey,
                state=transformer_state,
                return_state=return_state
            )
            if return_state:
                embeddings, new_transformer_state = embeddings
                new_transformer_states.append(new_transformer_state)

        normed_embeddings = self.final_layer_norm(embeddings)
        logits = self.embedder.unembed(normed_embeddings)

        if return_state:
            new_state = Gpt2State(
                transformer_states=new_transformer_states,
                first_position=first_position + 1 # Add 1 because of new logits
            )
            return logits, new_state
        else:
            return logits
        