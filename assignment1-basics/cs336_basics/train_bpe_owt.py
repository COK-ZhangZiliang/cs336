"""
Training BPE tokenizer on the TinyStories datasets
"""
from bpe import BPETokenizer

if __name__ == "__main__":
    data_path = "/volume/pt-train/users/wzhang/zzl-workspace/cs336/assignment1-basics/data/owt_train.txt"
    tokenizer = BPETokenizer()
    vocab, merges = tokenizer.train(
        input_path=data_path,
        vocab_size=32000
    )

    # Save the vocabulary and merges
    with open("/volume/pt-train/users/wzhang/zzl-workspace/cs336/assignment1-basics/cs336_basics/owt-vocab.txt", "w") as vocab_file:
        for idx, token in vocab.items():
            vocab_file.write(f"{idx}\t{token}\n")
    with open("/volume/pt-train/users/wzhang/zzl-workspace/cs336/assignment1-basics/cs336_basics/owt-merges.txt", "w") as merges_file:
        for pair in merges:
            merges_file.write(f"{pair[0]} {pair[1]}\n")
    
    # Find longest token
    longest_token = max(vocab.values(), key=len)
    print(longest_token, len(longest_token))
