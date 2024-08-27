# GPT2-Haliax
The goal of this project is to write a clean implementation of GPT 2 in Haliax.

## TODO
- [x] get inference working
- [ ] speed up inference (right now it is recomputing attention for everything every time--instead, it should only compute it for the next token)
- [ ] clean up positional axis logic in `run/infer.py`` (waiting for Haliax to implement anonymous axes...)
- [x] Add dropout layers where appropriate
- [ ] Add training (right now it only has inference)
- [ ] Figure out a better way to load safetensors/do serialization? See `run/load_gpt2` and `run/gpt2_skeleton` for how it is currently done.