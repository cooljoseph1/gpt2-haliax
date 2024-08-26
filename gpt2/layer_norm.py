from typing import NamedTuple

class LayerNormConfig(NamedTuple):
    eps: float = 0.00001
    use_weight: bool = True
    use_bias: bool = True

from haliax.nn.normalization import LayerNorm # alias
