"""
BPE Tokenizer Implementation by Python
BPE class functions: 
1. Init a BPETokenizer class; 
2. Using data to train a BPE tokenizer;
"""
import os
import regex as re
from collections import defaultdict

class BPETokenizer:
    def __init__(self):
        self.vocab: dict[int, bytes] = {}  # Token ID -> Token Bytes
        self.merges: list[tuple[bytes, bytes]] = []  # (First Symbols, Second Symbols)
        self.word_freq: defaultdict[int, int] = defaultdict(int)  # Index -> Word Frequency
        self.index_word: dict[int, list[bytes]] = {}  # Index -> Words
        self.word_index: dict[str, int] = {}  # Words -> Index
        self.pair_ids: defaultdict[tuple[bytes, bytes], set[int]] = defaultdict(set)  # (First Symbols, Second Symbols) -> Set of Indices
        self.pair_freq: defaultdict[tuple[bytes, bytes], int] = defaultdict(int)  # (First Symbols, Second Symbols) -> Pair Frequency
    
    def _split_by_special_tokens(
            self, 
            text: str, 
            special_tokens: list[str]
        ) -> list[str]:
        escaped_tokens = [re.escape(tok) for tok in special_tokens]
        pattern = f"{'|'.join(escaped_tokens)}"  # skip special tokens
        return [s for s in re.split(pattern, text) if s]
    
    def _add_pair(
            self,
            pair: tuple[bytes, bytes],
            idx: int,
            freq: int,
    ) -> None:
        self.pair_freq[pair] += freq
        self.pair_ids[pair].add(idx)

    def _remove_pair(
            self,
            pair: tuple[bytes, bytes],
            idx: int,
            freq: int,
    ) -> None:
        self.pair_freq[pair] -= freq
        if self.pair_freq[pair] <= 0:
            self.pair_freq.pop(pair, None)
            self.pair_ids.pop(pair, None)
        else:
            self.pair_ids[pair].discard(idx)

    def train(
            self, 
            input_path: str | os.PathLike,
            vocab_size: int,
            special_tokens: list[str]
        ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """
        Using text data to train a BPE tokenizer.
        """
        # 1) pre-tokenization
        with open(input_path, 'r') as f:
            data = f.read()
            chunks = self._split_by_special_tokens(data, special_tokens)
            PAT =  r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            for chunk in chunks:
                for match in re.finditer(PAT, chunk):
                    word = match.group(0)
                    word_bytes_rep = [bytes([b]) for b in word.encode('utf-8')]
                    if word in self.word_index:
                        idx = self.word_index[word]
                        self.word_freq[idx] += 1
                    else:
                        idx = len(self.index_word)
                        self.index_word[idx] = word_bytes_rep
                        self.word_index[word] = idx
                        self.word_freq[idx] += 1
        
        # 2) initial pair frequency & pair to ids
        for idx, word in self.index_word.items():
            freq = self.word_freq[idx]
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                self._add_pair(pair, idx, freq)


        # 3) merges
        num_merges = vocab_size - len(special_tokens) - 256  # Calculate number of merges needed
        for num in range(num_merges):
            # If two pairs have the same frequency, 
            # choose lexicographically greater one.
            # But the lexicographical order is decided by the first pair and then the second pair, 
            # rather than the concatenation of two pairs!!!
            pair = max(self.pair_freq, key=lambda x: (self.pair_freq[x], x))  
            a = self.pair_freq[pair]
            self.merges.append(pair)
            symbol_a, symbol_b = pair
            new_symbol = symbol_a + symbol_b
            affected_ids = list(self.pair_ids.get(pair, set()))  # Get affected indices

            for idx in affected_ids:
                word = self.index_word[idx]
                freq = self.word_freq[idx]
                new_word = []  # New word after merging
                i = 0
                while i < len(word) - 1:
                    if word[i] == symbol_a and word[i + 1] == symbol_b:
                        new_word.append(new_symbol)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                if i == len(word) - 1:
                    new_word.append(word[-1])  # Append the last symbol
                # Operate removal and addition separately to avoid unexpected errors
                for j in range(len(word) - 1):  # Remove old pairs
                    self._remove_pair((word[j], word[j + 1]), idx, freq)
                for j in range(len(new_word) - 1):  # Add new pairs
                    self._add_pair((new_word[j], new_word[j + 1]), idx, freq)
                self.index_word[idx] = new_word

        # 4) build vocabulary
        self.vocab = {i : special_tokens[i].encode('utf-8') for i in range(len(special_tokens))}
        self.vocab.update({i + len(special_tokens): bytes([i]) for i in range(256)})
        self.vocab.update({i + len(special_tokens) + 256: pair[0] + pair[1] for i, pair in enumerate(self.merges)})

        return self.vocab, self.merges
