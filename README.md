# GPT2-Haliax
The goal of this project is to write a clean implementation of GPT 2 in Haliax.

## TODO
- [x] get inference working
- [x] speed up inference (right now it is recomputing attention for everything every time--instead, it should only compute it for the next token)
- [x] Added `jax.jit` to make some things faster.
- [x] clean up positional axis logic in `run/infer.py``
- [x] Add dropout layers where appropriate
- [ ] Add training in a train/ folder (right now it has inference in a run/ folder)
- [ ] Figure out a better way to load safetensors/do serialization? See `run/load_gpt2` and `run/gpt2_skeleton` for how it is currently done.