from typing import Iterable, Iterator, Callable
import pickle
from cs336_basics.bpe import ENCODING, PAT
import regex as re


class Token:
    def __init__(self, data: bytes, next=None):
        self.data = data
        self.next: Token = next

    def __repr__(self) -> str:
        return f"Token({self.data=})"


class MemoizeDict(dict):
    def __init__(self, func: Callable):
        super().__init__()
        assert callable(func)
        self.func = func

    def __missing__(self, key):
        result = self.func(key)
        self[key] = result
        return result


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.vocab_reverse = {token: i for i, token in vocab.items()}
        self.merges = merges

        if special_tokens:
            self.special_tokens = sorted(special_tokens, reverse=True)
        else:
            self.special_tokens = None
        self.special_token_to_id = self._compute_special_token_to_id()

        self.words2ids_cache: MemoizeDict[bytes, list[int]] = MemoizeDict(
            func=self.word2ids
        )

    def _compute_special_token_to_id(self) -> dict[str, int]:
        special_token_to_id: dict[str, int] = dict()
        if self.special_tokens is None:
            return special_token_to_id
        special_tokens_bytes = set(
            t.encode(encoding=ENCODING) for t in self.special_tokens
        )
        for i, token_bytes in self.vocab.items():
            if token_bytes in special_tokens_bytes:
                token_str = token_bytes.decode(encoding=ENCODING)
                special_tokens_bytes.remove(token_bytes)
                special_token_to_id[token_str] = i
        for token_bytes in special_tokens_bytes:
            token_str = token_bytes.decode(encoding=ENCODING)
            self.vocab[len(self.vocab)] = token_bytes
            special_token_to_id[token_str] = len(self.vocab) - 1
        assert len(special_token_to_id) == len(self.special_tokens)
        return special_token_to_id

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[bytes] | None = None,
    ) -> "Tokenizer":
        with open(vocab_filepath, "rb") as file:
            vocab = pickle.load(file)
        with open(merges_filepath, "rb") as file:
            merges = pickle.load(file)
        return Tokenizer(vocab, merges, special_tokens)

    def word2ids(self, word: bytes) -> list[int]:
        prev: Token = None
        first_token: Token = None
        for i in word:
            token = Token(data=bytes([i]))
            if prev:
                prev.next = token
            else:
                first_token = token
            prev = token

        for merge in self.merges:
            token = first_token
            while token and token.next:
                if (token.data, token.next.data) == merge:
                    token.data = token.data + token.next.data
                    token.next = token.next.next
                token = token.next

        ids = []
        token = first_token
        while token:
            ids.append(self.vocab_reverse[token.data])
            token = token.next

        return ids

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        Assume that `text` fits into memory.
        """
        if self.special_tokens:
            # remove special tokens before pre-tokenization
            pattern = "|".join(re.escape(token) for token in self.special_tokens)
            # so that we keep the special tokens in the split
            chunks = re.split(f"({pattern})", text)
        else:
            chunks = [text]

        ids: list[int] = []
        for chunk in chunks:
            if chunk in self.special_token_to_id:
                ids += [self.special_token_to_id[chunk]]
            else:
                for match in re.finditer(PAT, chunk):
                    word: bytes = match.group().encode(encoding=ENCODING)
                    ids += self.words2ids_cache[word]
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return
        a generator that lazily yields token IDs. This is required for
        memory-eﬀicient tokenization of large files that we cannot directly
        load into memory.
        """
        buffer = ""
        for text in iterable:
            buffer += text
            if len(buffer) > 100_000_000:
                matches = list(re.finditer(PAT, buffer))
                assert len(matches) > 0
                last_match_end = matches[-1].end()
                to_process = buffer[:last_match_end]
                buffer = buffer[last_match_end:]
                yield from self.encode(to_process)
        yield from self.encode(buffer)

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        x = b"".join(self.vocab[id] for id in ids)
        x = x.decode(encoding=ENCODING, errors="replace")
        return x
