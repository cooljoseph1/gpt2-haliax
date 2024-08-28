# GPT2-Haliax
The goal of this project is to write a clean implementation of GPT 2 in Haliax.

## Requirements
This project requires Python 3.10. It uses JAX, Equinox, and Haliax for running the neural network,
and Pytree2Safetensors for loading the weights.

## Installation
First, clone this repository:
```sh
git clone git@github.com:cooljoseph1/gpt2-haliax.git
```

Next, go to the newly created directory:
```sh
cd gpt2-haliax
```

Then, I recommend setting up a virtual environment. If you have conda installed (either Miniconda or Anaconda), you
can do this with
```sh
conda create -n gpt2-haliax python=3.10
```
Your Python version needs to be at least Python 3.10.

Finally, install the requirements:
```
pip install -r requirements.txt
```

## Running
To do inference, run the command
```sh
python3 inference.py --prompt "<prompt>"
```
where `<prompt>` is your text prompt. There are more options, which can be printed out using the `--help` flag.

You don't have to provide a prompt; if you don't provide a prompt, it will instead read standard input for the prompt.

## TODO
- [ ] Get inference to go longer than 1024 tokens. (GPT2 was only trained with 1024 positional embeddings. This might not be possible to do efficiently.)
- [ ] Add training in a train/ folder (right now it has inference in a run/ folder)
- [ ] Figure out a better way to load safetensors? Right now I'm using Pytree2Safetensors, which is a not-very-polished library I made in a few hours.