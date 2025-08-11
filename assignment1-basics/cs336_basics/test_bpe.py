import unittest
import os
import tempfile
from collections import defaultdict

class TestBPETokenizer(unittest.TestCase):
    def setUp(self):
        """测试前的准备工作"""
        from bpe import BPETokenizer
        self.tokenizer = BPETokenizer()
        self.test_file = "/Users/ziliang/project/cs336/assignment1-basics/tests/fixtures/corpus.en"

    def test_train(self):
        """测试训练过程"""
        special_tokens = ["<|endoftext|>"]
        vocab_size = 500
        
        vocab, merges = self.tokenizer.train(
            input_path=self.test_file,
            vocab_size=vocab_size,
            special_tokens=special_tokens
        )

if __name__ == '__main__':
    unittest.main()