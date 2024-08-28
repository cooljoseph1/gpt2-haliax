import sys
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="""
        Run GPT2 inference using JAX and Haliax.
        It reads a prompt from stdin (unless --prompt is supplied) and prints the output to stdout.
        """
    )
    parser.add_argument("--prompt", type=str, default="", help="The prompt. If not provided, the program will read from stdin")
    parser.add_argument("--num-tokens", type=int, default=-1, help="The number of tokens to generate. Use -1 for an infinite number of tokens")
    parser.add_argument("--seed", type=int, default=0, help="The seed to the random number generator")
    
    args = parser.parse_args()

    prompt, num_tokens, seed = args.prompt, args.num_tokens, args.seed

    if prompt == "":
        prompt = sys.stdin.read()
    sys.stdin.close() # Not needed anymore--can close the pipe

    
    if prompt == "": # If no prompt is given, either as an arg or from stdin, print out the help and exit with an error
        parser.print_help(file=sys.stderr) # Print to stderr because it is an *error* to not provide a prompt
        sys.exit(2)
    

    import jax
    from .infer import infer_bytes
    
    for token_bytes in infer_bytes(
        prompt,
        num_tokens=num_tokens,
        key=jax.random.key(seed)
    ):
        sys.stdout.buffer.write(token_bytes)
        sys.stdout.flush()