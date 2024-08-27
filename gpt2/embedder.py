import jax
import jax.random as jrandom
import equinox as eqx
import haliax as hax
from haliax import Axis, AxisSelector, AxisSpec, NamedArray

from typing import NamedTuple

class EmbedderConfig(NamedTuple):
    max_position: int
    embed_size: int
    token_embedding_range: float = 0.02
    position_embedding_range: float = 0.01


class Embedder(eqx.Module):
    """Embed tokens to vectors"""

    EmbedAxis: AxisSpec = eqx.static_field()

    position_embedder: hax.nn.Embedding
    token_embedder: hax.nn.Embedding

    @staticmethod
    def init(
        VocabAxis: Axis,
        *,
        config: EmbedderConfig,
        key: jax.random.PRNGKey,
    ) -> "Embedder":
        
        EmbedAxis = Axis("embed", config.embed_size)
        
        k_position_embeddings, k_token_embeddings = jrandom.split(key, 2)
        
        position_embedder = hax.nn.Embedding.init(
            Vocab=Axis("position", config.max_position),
            Embed=EmbedAxis,
            initializer_range=config.position_embedding_range,
            key=k_position_embeddings
        )

        token_embedder = hax.nn.Embedding.init(
            Vocab=VocabAxis,
            Embed=EmbedAxis,
            initializer_range=config.token_embedding_range,
            key=k_token_embeddings
        )

        return Embedder(
            EmbedAxis=EmbedAxis,
            position_embedder=position_embedder,
            token_embedder=token_embedder
        )

    @jax.named_call
    def embed(
        self,
        input_ids: NamedArray,
        PositionAxis: AxisSelector = "position",
        *,
        first_position: int = 0 # Position of the first input_ids in the sequence. Not always zero, such as when caching values during inference
    ) -> NamedArray:
        # Positional embeddings
        PositionAxis = input_ids.resolve_axis(PositionAxis) # Convert to an Axis (perhaps it was a string)
        positions = hax.arange(PositionAxis, start=first_position)
        position_embeds = self.position_embedder(positions)

        # Text embeddings
        input_embeds = self.token_embedder(input_ids)

        result = input_embeds + position_embeds # WTF? Why add embeddings instead of concatenating?? GPT2 is weird...

        return result

    def unembed(
        self,
        embeddings: NamedArray
    ) -> NamedArray:
        return hax.dot(embeddings, self.token_embedder.weight, axis=self.EmbedAxis)