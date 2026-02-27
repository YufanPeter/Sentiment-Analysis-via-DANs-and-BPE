import re
from collections import Counter, defaultdict
from utils import Indexer
from sentiment_data import read_sentiment_examples

class BPE:
    """
    Byte Pair Encoding (BPE) Tokenizer.
    
    Implements the BPE algorithm to learn subword tokenization from a training corpus.
    It iteratively merges the most frequent adjacent pairs of characters/subwords until
    a target vocabulary size is reached.
    """
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.indexer = Indexer()
        self.merges = {}
        self.freqs = defaultdict(int)
        self.cache = {}

    def get_stats(self, vocab: dict):
        pairs = defaultdict(int)
        for word, freqs in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freqs
        return pairs
    
    def merge_vocab(self, pair: dict, vocab: dict):
        v_out = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in vocab:
            w_out = p.sub(''.join(pair), word)
            v_out[w_out] = vocab[word]
        return v_out
    
    def fit(self, train_file):
        examples = read_sentiment_examples(train_file)
        counter = Counter()
        for ex in examples:
            for word in ex.words:
                counter[word] += 1

        vocab = {" ".join(list(word)) + " </w>": freq for word, freq in counter.items()}
        self.indexer.add_and_get_index("PAD")
        self.indexer.add_and_get_index("UNK")

        chars = set()
        for word in vocab:
            for char in word.split():
                chars.add(char)
        for char in sorted(list(chars)):
            self.indexer.add_and_get_index(char)

        num_merges = self.vocab_size - len(self.indexer)

        for i in range(num_merges):
            pairs = self.get_stats(vocab)
            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get) # type: ignore
            self.merges[best_pair] = "".join(best_pair)

            new_token = "".join(best_pair)
            self.indexer.add_and_get_index(new_token)
            vocab = self.merge_vocab(best_pair, vocab)
            
            if (i + 1) % 100 == 0:
                print(f"BPE: {i + 1}/{num_merges} ...")

        self.vocab_mapping = vocab

        print("Pre-compiling merge rules")
        self.compiled_merges = []
        
        for pair, merged in self.merges.items():
            bigram = re.escape(' '.join(pair))
            p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
            self.compiled_merges.append((p, merged))

        print("Cache building")
        for segmented_word in vocab.keys():
            raw_word = segmented_word.replace(" ", "").replace("</w>", "")
            self.cache[raw_word] = segmented_word

    def encode(self, word):
        if word in self.cache:
            segmented_word = self.cache[word]
            tokens = segmented_word.split()

            ids = []
            for token in tokens:
                idx = self.indexer.index_of(token)
                if idx == -1:
                    idx = self.indexer.index_of("UNK")
                ids.append(idx)
            return ids

        current_word = " ".join(list(word)) + " </w>"
        
        for p, merged in self.compiled_merges:
            current_word = p.sub(merged, current_word)
        tokens = current_word.split()

        ids = []
        for token in tokens:
            idx = self.indexer.index_of(token)
            if idx == -1:
                idx = self.indexer.index_of("UNK")
            ids.append(idx)
        
        return ids