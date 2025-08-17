from typing import Iterable, Iterator
import pickle
from cs336_basics.bpe import ENCODING, PAT
import regex as re


class Token:
    def __init__(self, data: bytes, id: int, next=None):
        """
        vocab[id] = data
        """
        self.data = data
        self.id = id
        self.next: Token = next

    def __repr__(self) -> str:
        return f"Token({self.data=})"


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        if special_tokens:
            self.special_tokens = sorted(special_tokens, reverse=True)
        else:
            self.special_tokens = None

        self.special_token_to_id = self._compute_special_token_to_id()
        self.single_byte_to_id = self._compute_single_byte_to_id()
        self.vocab_merge_offset = self._compute_vocab_merge_offset()
        self._validate_vocab_merge_consistency()

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

    def _compute_single_byte_to_id(self) -> dict[bytes, int]:
        single_byte_to_id: dict[bytes, int] = dict()
        for i in range(2**8):
            single_byte = self.vocab[i]
            assert len(single_byte) == 1
            single_byte_to_id[single_byte] = i
        return single_byte_to_id

    def _compute_vocab_merge_offset(self) -> int:
        """
        We assume that during BPE, once merge starts, the vocab
        and merges will be updated in lockstep.
        """
        first_merge_data = b"".join(self.merges[0])
        for i_vocab in range(len(self.vocab)):
            if self.vocab[i_vocab] == first_merge_data:
                return i_vocab
        raise RuntimeError("Did not find vocab_merge_offset!")

    def _validate_vocab_merge_consistency(self) -> None:
        for i_merge, merge in enumerate(self.merges):
            i_vocab = self.vocab_merge_offset + i_merge
            assert self.vocab[i_vocab] == b"".join(merge)

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
            data = bytes([i])
            token = Token(data, id=self.single_byte_to_id[data])
            if prev:
                prev.next = token
            else:
                first_token = token
            prev = token

        for i_merge, merge in enumerate(self.merges):
            i_vocab = i_merge + self.vocab_merge_offset
            token = first_token

            is_first_token = True
            while token and token.next:
                if (token.data, token.next.data) == merge:
                    token.data = token.data + token.next.data
                    token.id = i_vocab
                    assert self.vocab[i_vocab] == b"".join(merge)
                    token.next = token.next.next
                if is_first_token:
                    first_token = token
                token = token.next
                is_first_token = False

        ids = []
        token = first_token
        while token:
            ids.append(token.id)
            token = token.next

        return ids

    def encode(
        self, text: str, *, word2ids_cache: dict[bytes, list[int]] = dict()
    ) -> list[int]:
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
                    if word in word2ids_cache:
                        ids += word2ids_cache[word]
                    else:
                        x = self.word2ids(word)
                        word2ids_cache[word] = x
                        ids += x
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return
        a generator that lazily yields token IDs. This is required for
        memory-eﬀicient tokenization of large files that we cannot directly
        load into memory.
        """
        word2ids_cache: dict[bytes, list[int]] = dict()
        buffer = ""
        for text in iterable:
            buffer += text
            while len(buffer) > 1_000_000:
                matches = list(re.finditer(PAT, buffer))
                assert len(matches) > 0
                last_match_end = matches[-1].end()
                to_process = buffer[:last_match_end]
                buffer = buffer[last_match_end:]
                yield from self.encode(to_process, word2ids_cache=word2ids_cache)
        yield from self.encode(buffer, word2ids_cache=word2ids_cache)

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        x = b"".join(self.vocab[id] for id in ids)
        x = x.decode(encoding=ENCODING, errors="replace")
        return x
