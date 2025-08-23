import argparse
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.bpe import ENCODING
import numpy as np


def main():
    """
    Example usage:
    [macbookpro 8/17/2025  9:56PM] ~/Library/CloudStorage/Dropbox/DL/cs336-assignment1 (main)
    $ python cs336_basics/scripts/tokenize_file.py output/tiny_stories_bpe_vocab.pkl output/tiny_stories_bpe_merges.pkl data/TinyStoriesV2-GPT4-valid.txt output/tiny_stories_valid_ids.py --special_tokens "<|endoftext|>"
    """
    parser = argparse.ArgumentParser(
        description="A script that takes a pretrained tokenizer and tokenize a text file."
    )

    parser.add_argument("vocab_filepath", type=str)
    parser.add_argument("merges_filepath", type=str)

    parser.add_argument("input_filepath", type=str)
    parser.add_argument("output_filepath", type=str)

    parser.add_argument("--special_tokens", type=str, nargs="*")

    args = parser.parse_args()

    tokenizer = Tokenizer.from_files(
        args.vocab_filepath, args.merges_filepath, args.special_tokens
    )

    tokenize_file(tokenizer, args.input_filepath, args.output_filepath)


def tokenize_file(
    tokenizer: Tokenizer, input_filepath: str, output_filepath: str
) -> None:
    chunk_idx = 0
    buffer: list[int] = []
    with (
        open(input_filepath, "r", encoding=ENCODING) as f_in,
        open(output_filepath, "wb") as f_out,
    ):
        for token_id in tokenizer.encode_iterable(f_in):
            buffer.append(token_id)
            if len(buffer) >= 1_000_000:
                chunk_idx += 1
                chunk = np.array(buffer, dtype=np.int16)
                chunk.tofile(f_out)
                buffer.clear()
                print(f"wrote the {chunk_idx}-th chunk to {output_filepath}")

        if buffer:
            chunk = np.array(buffer, dtype=np.int16)
            chunk.tofile(f_out)
            buffer.clear()

    # we can read the output file using:
    # final_ids = np.fromfile(output_filepath, dtype=np.int16)


if __name__ == "__main__":
    main()
