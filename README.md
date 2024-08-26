# GPT2-Haliax
The goal of this project is to write a clean implementation of GPT 2 in Haliax.

## TODO
- [x] inference works
- [ ] speed up inference (I'm not sure why, but it's using a ton of memory and is pretty slow)
- [ ] clean up positional axis logic (waiting for Haliax to implement anonymous axes...)
- [ ] Add dropout layers where appropriate
- [ ] Add training (right now it only has inference)
- [ ] Figure out a better way to load safetensors/do serialization? See `run/load_gpt2` and `run/gpt2_skeleton` for how it is currently done.