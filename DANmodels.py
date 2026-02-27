from re import sub
from altair import Data
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sentiment_data import read_sentiment_examples

class SentimentDatasetDAN(Dataset):
    """
    Dataset for the standard Deep Averaging Network (DAN) using GloVe embeddings.
    
    This dataset converts raw sentences into lists of integer indices corresponding 
    to words in the pre-trained embedding vocabulary.
    """
    def __init__(self, infile, wordembeddings):
        self.examples = read_sentiment_examples(infile)
        self.word_embeddings = wordembeddings

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        example = self.examples[index]

        indexer = self.word_embeddings.word_indexer
        unk_idx = indexer.index_of("UNK")

        indices = []
        for word in example.words:
            word_idx = indexer.index_of(word)
            if word_idx == -1:
                indices.append(unk_idx)
            else:
                indices.append(word_idx)

        return torch.tensor(indices, dtype=torch.long), torch.tensor(example.label, dtype=torch.long)

class SentimentDatasetBPE(Dataset):
    """
    Dataset for the BPE-based DAN model.
    
    This dataset converts sentences into lists of *subword* indices using 
    a trained Byte Pair Encoding (BPE) tokenizer.
    """
    def __init__(self, infile, bpe_tokenizer):
        self.examples = read_sentiment_examples(infile)
        self.bpe = bpe_tokenizer
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]

        subword_ids = []
        for word in example.words:
            ids = self.bpe.encode(word)
            subword_ids.extend(ids)

        return torch.tensor(subword_ids, dtype=torch.long), torch.tensor(example.label, dtype=torch.long)
    
class DAN(nn.Module):
    """
    Deep Averaging Network (DAN) Architecture.
    
    1. Embedding Layer: Converts indices to dense vectors.
    2. Averaging: Computes the mean of all word vectors in the sentence.
    3. Classifier: A multi-layer perceptron (MLP) that maps the averaged vector to class logits.
    """
    def __init__(self, embedding_layer, hidden_dim=100, ouput_dim=2, dropout=0.5):
        super(DAN, self).__init__()
        self.embeddings = embedding_layer
        embed_dim = self.embeddings.embedding_dim

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, ouput_dim)
        )

        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, indices):
        embeddings = self.embeddings(indices)
        vec = embeddings.mean(dim=1)
        logits = self.classifier(vec)
        
        return self.log_softmax(logits)