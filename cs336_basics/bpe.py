import os
import regex as re
from collections import defaultdict, Counter
from typing import Any, Iterable
from multiprocessing import Pool

from utils import log
import os
from typing import BinaryIO


ENCODING = "utf-8"
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def init_vocab(special_tokens: list[str]) -> list[bytes]:
    """
    Initialize the vocabulary with all 256 byte values and the special tokens.
    """
    vocab: list[bytes] = [bytes([i]) for i in range(2**8)]
    for token in special_tokens:
        vocab.append(token.encode(encoding=ENCODING))
    return vocab


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes | None,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()

    # Cannot split if no token is given.
    if split_special_token is None:
        return [0, file_size]

    assert isinstance(
        split_special_token, bytes
    ), "Must represent special token as a bytestring"

    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def read_chunks(
    input_path: str | os.PathLike, special_tokens: list[str], start: int, end: int
) -> list[str]:
    """
    - read the file from start to end
    - split them into a list of chunks using the special tokens.
    """
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk: str = f.read(end - start).decode(encoding=ENCODING)

    if special_tokens:
        # remove special tokens before pre-tokenization
        pattern = "|".join(re.escape(token) for token in special_tokens)
        return re.split(pattern, chunk)
    else:
        return [chunk]


def count_words(chunks: list[str]) -> dict[bytes, int]:
    """
    We split the chunks using the GPT-2 PAT regex and count words.
    """
    word_frequency: dict[tuple[bytes, ...], int] = defaultdict(int)
    for chunk in chunks:
        # there is no special token in chunk
        for match in re.finditer(PAT, chunk):
            word: bytes = match.group().encode(encoding=ENCODING)
            word_frequency[word] += 1
    return word_frequency


def read_chunks_and_count_words(*_args):
    return count_words(read_chunks(*_args))


def add_dicts(list_of_dict: Iterable[dict[Any, int]]) -> dict[Any, int]:
    """
    A helper function to add a few dictionaries.
    """
    total = Counter()
    for d in list_of_dict:
        total.update(d)
    return defaultdict(int, total)


def read_file_and_count_words(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    num_chunks: int,
    num_processes: int,
) -> dict[bytes, int]:
    """
    Split the text using the special tokens and the regex pattern PAT,
    and then count the frequency of words. We speed up by first spliting
    the text into a few chunks of roughly the same size. Then we
    multiprocess the chunks in parallel.
    """
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f,
            num_chunks,
            special_tokens[0].encode(encoding=ENCODING) if special_tokens else None,
        )

    with Pool(processes=num_processes) as pool:
        args = [
            (input_path, special_tokens, start, end)
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]
        list_of_word_frequency: list[dict[bytes, int]] = pool.starmap(
            read_chunks_and_count_words, args
        )
    return add_dicts(list_of_word_frequency)


def compute_pair_stats(
    word_freq: dict[bytes, int],
) -> tuple[
    dict[bytes, list[bytes]],
    dict[tuple[bytes, bytes], tuple[int, dict[bytes, int]]],
]:
    """
    Returns:
        how_to_split dict[bytes, list[bytes]]:
            how_to_split[word] gives a list of bytes whose composition is equal to word
            this is our current way of tokenizing the given word
        pair_stats dict[tuple[bytes, bytes], tuple[int, dict[bytes, int]]]:
            pair_stats[pair][0] is the frequency of this pair in the text
            pair_stats[pair][1][word] is the number of times this pair shows up in word
    """
    # how to split a word in the current vocab
    # e.g. we might split "abc" into "a" and "bc"
    how_to_split: dict[bytes, list[bytes]] = dict()

    # int is pair frequency
    # dict[bytes, int] maps from
    #   the word in which the pair shows up in
    #       to
    #   the number of times this pair shows up in the word
    pair_stats: dict[
        tuple[bytes, bytes],
        tuple[int, dict[bytes, int]],
    ] = dict()

    for word, freq in word_freq.items():
        split = [bytes([i]) for i in word]
        how_to_split[word] = split
        for pair in zip(split[:-1], split[1:]):
            add_pair(pair, word, freq, pair_stats)
    return how_to_split, pair_stats


def add_pair(
    pair: tuple[bytes, bytes],
    word: bytes,
    freq: int,
    pair_stats: dict[
        tuple[bytes, bytes],
        tuple[int, dict[bytes, int]],
    ],
) -> None:
    """
    Helper function.
    Inputs:
        freq = the frequency of word in the text
    """
    if pair not in pair_stats:
        pair_stats[pair] = (freq, defaultdict(int, {word: 1}))
    else:
        old_freq, old_dict = pair_stats[pair]
        new_freq = old_freq + freq
        old_dict[word] += 1
        pair_stats[pair] = new_freq, old_dict


def del_pair(
    pair: tuple[bytes, bytes],
    word: bytes,
    freq: int,
    pair_stats: dict[
        tuple[bytes, bytes],
        tuple[int, dict[bytes, int]],
    ],
) -> None:
    """
    Helper function.
    Inputs:
        freq = the frequency of word in the text
    """
    old_freq, old_dict = pair_stats[pair]

    new_freq = old_freq - freq
    old_dict[word] -= 1
    if old_dict[word] == 0:
        old_dict.pop(word)

    if new_freq == 0:
        assert len(old_dict) == 0
        pair_stats.pop(pair)
    else:
        assert len(old_dict) > 0
        pair_stats[pair] = new_freq, old_dict


def resplit(
    word: bytes,
    word_freq: int,
    pair: tuple[bytes, bytes],
    how_to_split: dict[bytes, list[bytes]],
    pair_stats: dict[
        tuple[bytes, bytes],
        tuple[int, dict[bytes, int]],
    ],
) -> None:
    # print(f"{word=}")
    # print(f"{pair=}")

    old_split = how_to_split[word]
    new_split, del_pairs, add_pairs = greedily_merge(old_split, pair)
    how_to_split[word] = new_split

    # print(f"{old_split=}")
    # print(f"{new_split=}")

    # print(f"{del_pairs=}")
    # print(f"{add_pairs=}")

    for del_pair, pairs_per_word in del_pairs.items():
        # print(f"{pair_stats=}")
        freq, old_dict = pair_stats.pop(del_pair)
        freq -= word_freq * pairs_per_word
        old_dict[word] -= pairs_per_word
        if old_dict[word] == 0:
            old_dict.pop(word)
        if len(old_dict) == 0:
            # assert len(old_dict) == 0, (del_pair, old_dict)
            continue
        pair_stats[del_pair] = freq, old_dict

    for add_pair, pairs_per_word in add_pairs.items():
        if add_pair in pair_stats:
            freq, old_dict = pair_stats[add_pair]
        else:
            freq, old_dict = 0, defaultdict(int)
        freq += word_freq * pairs_per_word
        old_dict[word] += pairs_per_word
        pair_stats[add_pair] = freq, old_dict


def greedily_merge(
    old_split: list[bytes],
    pair: tuple[bytes, bytes],
) -> tuple[
    list[bytes],
    dict[tuple[bytes, bytes], int],
    dict[tuple[bytes, bytes], int],
]:
    """
    Greedily merge all pairs in the old split.
    Also computes how many pairs get deleted and how many pairs get added.

    For example, if
        old_split = (a, b, c, b, c)
        pair = (b, c)
    then
        new_split = (a, bc, bc, d),
        del_pairs = {(a, b): 1, (b, c): 2, (c, b): 1}
        new_pairs = {(a, bc): 1, (bc, bc): 1, (bc, d): 1}
    """
    new_split = []
    del_pairs: dict[tuple[bytes, bytes], int] = defaultdict(int)
    add_pairs: dict[tuple[bytes, bytes], int] = defaultdict(int)

    # shining
    # del: hi, ni, ng
    # add: hin, inin, ing

    i = 0
    just_merged = False
    while i < len(old_split):
        if i + 1 < len(old_split) and (old_split[i], old_split[i + 1]) == pair:
            if i - 1 >= 0:
                del_pairs[(old_split[i - 1], old_split[i])] += 1
            if len(new_split) > 0:
                add_pairs[(new_split[-1], b"".join(pair))] += 1
            new_split.append(b"".join(pair))
            i += 2
            just_merged = True
        else:
            if just_merged:
                del_pairs[(old_split[i - 1], old_split[i])] += 1
                add_pairs[(new_split[-1], old_split[i])] += 1
            new_split.append(old_split[i])
            i += 1
            just_merged = False
    return new_split, del_pairs, add_pairs


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    *,
    num_chunks: int = 100,
    num_processes: int = 8,
    log_enabled: bool = False,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.
        num_chunks (int): We first split the file into roughly these many chunks,
            so that each chunk fits in RAM memory and we count word frequency of
            each chunk separately.
        num_processes (int): We parallelize pre-tokenization by multiprocessing.
        log_enabled (bool): Whether to print timing and progress information while running.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab_dict:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    with log("init vocab", log_enabled):
        vocab_list = init_vocab(special_tokens)

    with log("find boundaries", log_enabled):
        word_freq = read_file_and_count_words(
            input_path,
            special_tokens,
            num_chunks=num_chunks,
            num_processes=num_processes,
        )

    with log(f"compute_token_pair_frequency", log_enabled):
        how_to_split, pair_stats = compute_pair_stats(word_freq)

    def _sort_kv_by(
        kv: tuple[
            tuple[bytes, bytes],
            tuple[int, dict[bytes, int]],
        ],
    ):
        pair, stats = kv
        freq, _ = stats
        return (freq, pair)

    merges: list[tuple[bytes, bytes]] = []
    while len(vocab_list) < vocab_size and len(pair_stats) > 0:
        pair, (_, words) = max(pair_stats.items(), key=_sort_kv_by)
        words = words.keys()

        vocab_list.append(b"".join(pair))
        merges.append(pair)

        for word in words:
            # update how_to_split and pair_stats
            resplit(word, word_freq[word], pair, how_to_split, pair_stats)

        pair_stats.pop(pair)

    vocab_dict: dict[int, bytes] = {i: b for i, b in enumerate(vocab_list)}
    return vocab_dict, merges
