"""
BPE Tokenizer Implementation by Python
BPE class functions: 
1. Init a BPETokenizer class; 
2. Using data to train a BPE tokenizer;
"""
import os
import regex as re
import logging
from collections import defaultdict, Counter
from typing import BinaryIO, Iterable, Iterator
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time, json, copy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BPETokenizer:
    def __init__(
            self, 
            vocab: dict[int, bytes] | None = None,
            merges: list[tuple[bytes, bytes]] | None = None,
            special_tokens: list[str] = ["<|endoftext|>"],
            trained: bool = False
    ) -> None:
        """
        Be careful to transfer a default value like "{}" via class methods, because when having multiple objects, 
        they will share the same content on changeable parameters that have default values.
        """
        self.vocab: dict[int, bytes] = vocab if vocab else {}  # Token ID -> Token Bytes
        self.merges: list[tuple[bytes, bytes]] = merges if merges else [] # (First Symbols, Second Symbols)
        self.word_to_freq: defaultdict[int, int] = defaultdict(int)  # Index -> Word Frequency
        self.idx_to_word: dict[int, list[bytes]] = {}  # Index -> Words
        self.pair_to_ids: defaultdict[tuple[bytes, bytes], set[int]] = defaultdict(set)  # (First Symbols, Second Symbols) -> Set of Indices
        self.pair_to_freq: defaultdict[tuple[bytes, bytes], int] = defaultdict(int)  # (First Symbols, Second Symbols) -> Pair Frequency
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""  # Regex pattern for pre-tokenization
        self.special_tokens = special_tokens

        if trained:
            self.idx_to_token = vocab
            self.token_to_idx = {tok: idx for idx, tok in vocab.items()}
            for sp_tok in special_tokens:
                sp_tok.encode("utf-8")
                if sp_tok not in self.token_to_idx:
                    idx = len(self.idx_to_token)
                    self.idx_to_token[idx] = sp_tok
                    self.token_to_idx[sp_tok] = idx
                    self.vocab[idx] = sp_tok
            self.merges_set = set(self.merges)



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
        self.pair_to_freq[pair] += freq
        self.pair_to_ids[pair].add(idx)


    def _remove_pair(
            self,
            pair: tuple[bytes, bytes],
            idx: int,
            freq: int,
    ) -> None:
        self.pair_to_freq[pair] -= freq
        if self.pair_to_freq[pair] <= 0:
            self.pair_to_freq.pop(pair, None)
            self.pair_to_ids.pop(pair, None)
        else:
            self.pair_to_ids[pair].discard(idx)


    def train(
            self, 
            input_path: str | os.PathLike,
            vocab_size: int,
            special_tokens: list[str] = ["<|endoftext|>"],
            split_token: str = "<|endoftext|>"
        ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """
        Using text data to train a BPE tokenizer.
        """
        # 1) pre-tokenization
        logger.info("Pre-tokenization started...")
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
        logger.info(f"Pre-tokenization completed in {time.time() - start_time:.2f} seconds.")
        logger.info("Building pre-tokens' frequency dictionary...")
        start_time = time.time()
        for i, (word, freq) in tqdm(
                enumerate(word_count.items()),
                total=len(word_count)
            ):
            self.idx_to_word[i] = [bytes([b]) for b in word.encode('utf-8')]
            self.word_to_freq[i] = freq
        logger.info(f'Pre-tokens frequency dictionary built in {time.time() - start_time:.2f} seconds.')
        
        # 2) initial pair frequency & pair to ids
        logger.info("Initial pair frequency calculation started...")
        start_time = time.time()
        for idx, word in tqdm(
                self.idx_to_word.items(),
                total=len(self.idx_to_word),
            ):  # Convert word to bytes and calculate pairs
            freq = self.word_to_freq[idx]
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                self._add_pair(pair, idx, freq)
        logger.info(f"Initial pair frequency calculation completed in {time.time() - start_time:.2f} seconds.")

        # 3) merges
        logger.info("Merging pairs started...")
        start_time = time.time()
        num_merges = vocab_size - len(special_tokens) - 256  # Calculate number of merges needed
        for num in tqdm(range(num_merges)):
            # If two pairs have the same frequency, 
            # choose lexicographically greater one.
            # But the lexicographical order is decided by the first pair and then the second pair, 
            # rather than the concatenation of two pairs!!!
            if not self.pair_to_freq:
                break
            pair = max(self.pair_to_freq, key=lambda x: (self.pair_to_freq[x], x))  
            self.merges.append(pair)
            symbol_a, symbol_b = pair
            new_symbol = symbol_a + symbol_b
            affected_ids = list(self.pair_to_ids.get(pair, set()))  # Get affected indices

            for idx in affected_ids:
                word = self.idx_to_word[idx]
                freq = self.word_to_freq[idx]
                new_word = []  # New word after merging
                i = 0
                while i < len(word) - 1:
                    if word[i] == symbol_a and word[i + 1] == symbol_b:
                        new_word.append(new_symbol)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_word += word[i:]  # Append the last symbol
                # Operate removal and addition separately to avoid unexpected errors
                for j in range(len(word) - 1):  # Remove old pairs
                    self._remove_pair((word[j], word[j + 1]), idx, freq)
                for j in range(len(new_word) - 1):  # Add new pairs
                    self._add_pair((new_word[j], new_word[j + 1]), idx, freq)
                self.idx_to_word[idx] = new_word
        logger.info(f"Merging pairs completed in {time.time() - start_time:.2f} seconds.")

        # 4) build vocabulary
        self.vocab = {i : special_tokens[i].encode('utf-8') for i in range(len(special_tokens))}
        self.vocab.update({i + len(special_tokens): bytes([i]) for i in range(256)})
        self.vocab.update({i + len(special_tokens) + 256: pair[0] + pair[1] for i, pair in enumerate(self.merges)})
        logger.info("Vocabulary built successfully.")

        # Init other variables for futher encode & decode
        self.special_tokens = special_tokens
        self.idx_to_token = self.vocab
        self.token_to_idx = {tok: idx for idx, tok in self.vocab.items()}
        self.merges_set = set(self.merges)

        return self.vocab, self.merges


    @classmethod
    def init_by_files(
            cls, 
            vocab_filepath: str | os.PathLike, 
            merges_filepath: str | os.PathLike, 
            special_tokens: list[str] = ["<|endoftext|>"]
        ) -> "BPETokenizer":
        """
        Using vocab file & merges file to initial a bpe tokenizer
        """
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            token_to_idx = json.loads(f)
        vocab = {}
        for token, idx in token_to_idx.items():
            vocab = {idx, token.encode("utf-8")}
        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                a, b = line.strip().split(" ")
                merges.append((a.encode("utf-8"), b.encode("utf-8")))

        return cls(vocab, merges, special_tokens=special_tokens, trained=True)
    

    def encode(self, text: str) -> list[int]:
        # pre-tokenizaiton
        b_words = []
        for match in re.finditer(self.PAT, text):
            word = match.group(0)   
            if word in self.special_tokens:
                b_words.append([word.encode('utf-8')])
            b_words.append([bytes[b] for b in word.encode('utf-8')])
        
        # merge
        for w in b_words:
            i = 0
            while i < len(w) - 1:
                if (w[i], w[i + 1]) in self.merges_set:
                    w[i:i+2] = [w[i] + w[i + 1]]
                    i = max(i - 1, 0)  # At most affect the previous one
                else:
                    i += 1
        
        return [self.token_to_idx[tok] for w in b_words for tok in w]
    

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for idx in self.encode(text):
                yield idx
    

    def decode(self, ids: list[int]) -> str:
        tokens = [self.idx_to_token[idx] for idx in ids]
        b_text = b"".join(tokens)
        return b_text.decode('utf-8', errors='replace')