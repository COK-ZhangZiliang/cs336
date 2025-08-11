"""
BPE Tokenizer Implementation by Python
BPE class functions: 
1. Init a BPETokenizer class; 
2. Using data to train a BPE tokenizer;
"""
import os
import regex as re
from collections import defaultdict, Counter
from typing import BinaryIO
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time

class BPETokenizer:
    def __init__(self):
        self.vocab: dict[int, bytes] = {}  # Token ID -> Token Bytes
        self.merges: list[tuple[bytes, bytes]] = []  # (First Symbols, Second Symbols)
        self.word_freq: defaultdict[int, int] = defaultdict(int)  # Index -> Word Frequency
        self.index_word: dict[int, list[bytes]] = {}  # Index -> Words
        self.pair_ids: defaultdict[tuple[bytes, bytes], set[int]] = defaultdict(set)  # (First Symbols, Second Symbols) -> Set of Indices
        self.pair_freq: defaultdict[tuple[bytes, bytes], int] = defaultdict(int)  # (First Symbols, Second Symbols) -> Pair Frequency
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""  # Regex pattern for pre-tokenization


    @staticmethod
    def _find_chunk_boundaries(
        file: BinaryIO, 
        desired_num_chunks: int, 
        split_special_token: bytes
    ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), (
            "Must represent special token as a bytestring"
        )

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
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


    @staticmethod    
    def _chunk_pretokenize(args) -> None:
        input_path, start, end, PAT, special_tokens= args
        c = Counter()
        escaped_tokens = [re.escape(tok) for tok in special_tokens]
        pattern = f"{'|'.join(escaped_tokens)}"  # skip special tokens
        with open(input_path, 'rb') as f:
            f.seek(start)
            data = f.read(end - start)
            split_data = [d for d in re.split(pattern, data.decode('utf-8')) if d]
            for d in split_data:
                for match in re.finditer(PAT, d):
                    word = match.group(0)
                    c[word] += 1
        return c

  
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
            special_tokens: list[str],
            split_token: str = "<|endoftext|>"
        ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """
        Using text data to train a BPE tokenizer.
        """
        # 1) pre-tokenization
        print("Pre-tokenization started...")
        start_time = time.time()
        num_processes = max(1, min(cpu_count(), 16))  # Limit to 16 processes
        with open(input_path, 'rb') as f:
            boundaries = self._find_chunk_boundaries(
                f, 
                num_processes,
                split_token.encode('utf-8')
            )
        tasks = []
        word_count = Counter()
        for s, e in zip(boundaries[:-1], boundaries[1:]):
            tasks.append((input_path, s, e, self.PAT, special_tokens))
        if tasks:
            with Pool(min(num_processes, len(tasks)), maxtasksperchild=64) as pool:  # Pre-tokenization in parallel
                for c in tqdm(
                    pool.imap_unordered(self._chunk_pretokenize, tasks, chunksize=1),
                    total = len(tasks)
                ):
                    word_count.update(c)  # Using local Counter to aggregate word frequencies
        print(f"Pre-tokenization completed in {time.time() - start_time:.2f} seconds.")
        print("Building pre-tokens' frequency dictionary...")
        start_time = time.time()
        for i, (word, freq) in tqdm(
                enumerate(word_count.items()),
                total=len(word_count)
            ):
            self.index_word[i] = [bytes([b]) for b in word.encode('utf-8')]
            self.word_freq[i] = freq
        print(f'Pre-tokens frequency dictionary built in {time.time() - start_time:.2f} seconds.')
        
        # 2) initial pair frequency & pair to ids
        print("Initial pair frequency calculation started...")
        start_time = time.time()
        for idx, word in tqdm(
                self.index_word.items(),
                total=len(self.index_word),
            ):  # Convert word to bytes and calculate pairs
            freq = self.word_freq[idx]
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                self._add_pair(pair, idx, freq)
        print(f"Initial pair frequency calculation completed in {time.time() - start_time:.2f} seconds.")

        # 3) merges
        print("Merging pairs started...")
        start_time = time.time()
        num_merges = vocab_size - len(special_tokens) - 256  # Calculate number of merges needed
        for num in tqdm(range(num_merges)):
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
        print(f"Merging pairs completed in {time.time() - start_time:.2f} seconds.")

        # 4) build vocabulary
        self.vocab = {i : special_tokens[i].encode('utf-8') for i in range(len(special_tokens))}
        self.vocab.update({i + len(special_tokens): bytes([i]) for i in range(256)})
        self.vocab.update({i + len(special_tokens) + 256: pair[0] + pair[1] for i, pair in enumerate(self.merges)})
        print("Vocabulary built successfully.")

        return self.vocab, self.merges
