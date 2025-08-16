import os
import regex as re
from collections import defaultdict, Counter
from typing import Any, Iterable
from multiprocessing import Pool
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


class Token:
    def __init__(self, data: bytes, freq: int, ord: int, prev=None, next=None):
        """
        freq: the frequency of the word to which this token belongs.
        ord: the position of this token in the word to which it belongs.
        """
        self.data = data
        self.freq: int = freq
        self.prev: Token = prev
        self.next: Token = next
        self.ord: int = ord

    def __repr__(self) -> str:
        return f"Token({self.data=}, {self.freq=}, {self.ord=})"


def init_pair_occu_freq(word_freq: dict[bytes, int]) -> tuple[
    dict[tuple[bytes, bytes], set[Token]],
    dict[tuple[bytes, bytes], int],
]:
    """
    Returns:
        pair_occu: dict[tuple[bytes, bytes], set[Token]]
        pair_freq: dict[tuple[bytes, bytes], int]
    """
    pair_occu: dict[tuple[bytes, bytes], set[Token]] = defaultdict(set)
    pair_freq: dict[tuple[bytes, bytes], int] = defaultdict(int)
    for word, freq in word_freq.items():
        prev: Token = None
        for ord, i in enumerate(word):
            data = bytes([i])
            token = Token(data, freq, ord=ord, prev=prev)
            if prev:
                prev.next = token
                pair_freq[(prev.data, data)] += freq
                pair_occu[(prev.data, data)].add(prev)
            prev = token
    return pair_occu, pair_freq


def get_most_common_pair(
    pair_freq: dict[tuple[bytes, bytes], int],
) -> tuple[bytes, bytes]:
    """
    This implementation is much faster than
        max(pair_freq.items(), key=lamda kv: (kv[1], kv[0]))[0]
    probably because comparing two bytes is slow.
    """
    candidates: set[tuple[bytes, bytes]] = set()
    max_freq = 0
    for pair, freq in pair_freq.items():
        if freq > max_freq:
            candidates = {pair}
            max_freq = freq
        elif freq == max_freq:
            candidates.add(pair)
    return max(candidates)


def merge_bc(
    token: Token,
    pair_occu: dict[tuple[bytes, bytes], set[Token]],
    pair_freq: dict[tuple[bytes, bytes], int],
):
    """
    Merge token with token.next.
    Update pair_occu and pair_freq accordingly.
    """
    # (a, b, c, d) -> (a, bc, d)

    b = token
    del token
    a = b.prev
    c = b.next
    assert c is not None
    d = c.next
    freq = b.freq

    pair_occu[(b.data, c.data)].remove(b)

    b.next = d
    if d is not None:
        d.prev = b

    # keep pair_freq and pair_occu up to date
    if a is not None:
        pair_freq[(a.data, b.data)] -= freq
        pair_occu[(a.data, b.data)].remove(a)
    if d is not None:
        pair_freq[(c.data, d.data)] -= freq
        pair_occu[(c.data, d.data)].remove(c)

    b.data = b"".join([b.data, c.data])

    if a is not None:
        pair_freq[(a.data, b.data)] += freq
        pair_occu[(a.data, b.data)].add(a)
    if d is not None:
        pair_freq[(b.data, d.data)] += freq
        pair_occu[(b.data, d.data)].add(b)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    *,
    num_chunks: int = 100,
    num_processes: int = 8,
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
    vocab_list = init_vocab(special_tokens)

    word_freq = read_file_and_count_words(
        input_path,
        special_tokens,
        num_chunks=num_chunks,
        num_processes=num_processes,
    )

    # pair_occu[bytes pair] = list[token: (token.data, token.next.data) = bytes pair]
    # pair_freq[bytes pair] = the frequency of this bytes pair in the whole text
    pair_occu, pair_freq = init_pair_occu_freq(word_freq)
    del word_freq

    merges: list[tuple[bytes, bytes]] = []
    while len(vocab_list) < vocab_size and len(pair_freq) > 0:
        pair = get_most_common_pair(pair_freq)
        del pair_freq[pair]

        # update vocab_list and merges with the selected pair
        new_vocab = b"".join(pair)
        vocab_list.append(new_vocab)
        merges.append(pair)

        # For several collapsible token pairs in the same word,
        # we always collapse the leftmost token pair first.
        sorted_tokens = sorted(pair_occu[pair], key=lambda x: x.ord)
        for token in sorted_tokens:
            if token in pair_occu[pair]:
                merge_bc(token, pair_occu, pair_freq)
        del pair_occu[pair]

    vocab_dict: dict[int, bytes] = {i: b for i, b in enumerate(vocab_list)}
    return vocab_dict, merges
