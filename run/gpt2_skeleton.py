from gpt2 import *
import haliax as hax
import jax
import equinox as eqx

gpt2_config = Gpt2Config(
    embedder_config=EmbedderConfig(
        embed_size=768,
        max_position=1024,
    ),
    num_layers=12,
    transformer_config=TransformerConfig(
        layer_norm1_config=LayerNormConfig(),
        layer_norm2_config=LayerNormConfig(),
        attention_config=AttentionConfig(
            num_heads=12,
            head_embed_size=768 // 12
        ),
        feed_forward_config=FeedForwardConfig(
            hidden_size=768 * 4
        )
    ),
    final_layer_norm_config=LayerNormConfig()
)

VocabAxis = hax.Axis("vocab", 50257)

gpt2_skeleton = eqx.filter_eval_shape(
    Gpt2.init,
    VocabAxis=VocabAxis,
    config=gpt2_config,
    key=jax.random.key(3)
)