import jax
import jax.random as jrandom
import equinox as eqx
import haliax as hax
from haliax import Axis, AxisSpec, NamedArray

from typing import NamedTuple

class FeedForwardConfig(NamedTuple):
    hidden_size: int
    use_bias: bool = True

class FeedForwardBlock(eqx.Module):
    """A single transformer feed forward block"""

    InAxis: AxisSpec = eqx.field(static=True)
    HiddenAxis: AxisSpec = eqx.field(static=True)
    OutAxis: AxisSpec = eqx.field(static=True)

    project_hidden: hax.nn.Linear  # projection from inputs to the hidden layer
    project_out: hax.nn.Linear  # projection from the hidden layer to the outputs

    @staticmethod
    def init(
        InAxis: AxisSpec,
        OutAxis: AxisSpec,
        *,
        config: FeedForwardConfig,
        key: jax.random.PRNGKey,
    ) -> "FeedForwardBlock":
        HiddenAxis = Axis("hidden", config.hidden_size)
        k_hidden, k_out = jrandom.split(key, 2)
        project_hidden = hax.nn.Linear.init(In=InAxis, Out=HiddenAxis, key=k_hidden, use_bias=config.use_bias)
        project_out = hax.nn.Linear.init(In=HiddenAxis, Out=OutAxis, key=k_out, use_bias=config.use_bias)

        return FeedForwardBlock(
            InAxis=InAxis,
            HiddenAxis=HiddenAxis,
            OutAxis=OutAxis,
            project_hidden=project_hidden,
            project_out=project_out
        )

    @jax.named_call
    def __call__(self, x: NamedArray) -> NamedArray:
        x = self.project_hidden(x)
        x = hax.nn.activations.gelu(x, approximate=False)
        x = self.project_out(x)
        return x