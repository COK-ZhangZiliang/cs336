"""
Training BPE tokenizer on the TinyStories datasets
"""
from bpe import BPETokenizer

if __name__ == "__main__":
    data_path = "/Users/ziliang/project/cs336/data/TinyStoriesV2-GPT4-train.txt"
    tokenizer = BPETokenizer()
    vocab, merges = tokenizer.train(
        input_path=data_path,
        vocab_size=10000
    )
    # Save the vocabulary and merges
    with open("/Users/ziliang/project/cs336/assignment1-basics/cs336_basics/vocab.txt", "w") as vocab_file:
        for idx, token in vocab.items():
            vocab_file.write(f"{idx}\t{token}\n")
    with open("/Users/ziliang/project/cs336/assignment1-basics/cs336_basics/merges.txt", "w") as merges_file:
        for pair in merges:
            merges_file.write(f"{pair[0]} {pair[1]}\n")
