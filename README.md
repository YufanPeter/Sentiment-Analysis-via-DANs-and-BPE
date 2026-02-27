# Sentiment Analysis via Deep Averaging Networks and BPE

## Author
Yufan Shi

## Overview
This repository explores different neural network architectures for sentiment analysis on movie review dataset. There are three model architectures:

1.  **Bag of Words (BOW)**: Baseline models (NN2 and NN3) using simple averaging.
2.  **Deep Averaging Network (DAN)**: A model that averages pre-trained GloVe word embeddings passed through a deep neural network.
3.  **Byte Pair Encoding (BPE)**: A custom BPE tokenizer trained from scratch to learn subword embeddings.

## Requirements
Please ensure you have the following libraries installed:
1. Python 3.x
2. PyTorch
3. NumPy
4. Scikit-learn
5. Matplotlib

## How to Run
### 1. Run Deep Averaging Network (DAN) Using GloVe and Random Initialization

This uses pre-trained GloVe embeddings. By default, it uses the **50d** vectors and a hidden size of **100**.

```bash
python main.py --model DAN
```

### 2. Run Byte Pair Encoding (BPE)

This trains a BPE tokenizer from the training data and learns embeddings from scratch and the default vocabbulary size is **2,000**.

```bash
python main.py --model BPE
```

## Configuration & Hyperparameters

Modify `main.py` or `DANmodels.py` directly to adjust network parameters and conduct ablation studies.

### 1. DAN Architecture Configuration
Navigate to the `elif args.model == "DAN":` block in `main.py`.

* **Switching Embeddings (50d vs 300d)**:
    Uncomment the desired pre-trained embedding file path:
    ```python
    # Default setting
    embeddings_file = 'data/glove.6B.50d-relativized.txt'
    # embeddings_file = 'data/glove.6B.300d-relativized.txt'
    ```

* **Random Initialization (Ablation Study)**:
    To evaluate the model learning from scratch without pre-trained GloVe embeddings:
    ```python
    USE_RANDOM = True  # Set to True for random initialization, False for GloVe Embeddings
    ```

* **Hidden Dimensions**:
    Modify the `hidden_dim` parameter during model instantiation:
    ```python
    dan_model = DAN(embedding_layer, hidden_dim=100) # e.g., Change 100 to 256
    ```
* **Network Depth**:
    To test a shallower network, modify the classifier in `DANmodels.py` by commenting out the intermediate hidden layers:
    ```python
    # nn.Linear(hidden_dim, hidden_dim),
    # nn.ReLU(),
    # nn.Dropout(dropout)
    ```
* **Regularization**:
    Adjust the `dropout` parameter in `DANmodels.py` to mitigate overfitting:
    ```python
    dropout = 0.5 # Default set to 0.5 for optimal 300d performance
    ```

### 2. BPE Tokenizer Configuration
Navigate to the `elif args.model == "BPE":` block in `main.py`.

* **Vocabulary Size**:
    Change the `vocab_size` variable to experiment with different tokenization granularities (e.g., investigating over-segmentation):
    ```python
    vocab_size = 2000 # Default is 2000; experiment with 10000 for balanced subwords
    ```

* **Embedding Dimensions**:
    ```python
    embed_dim = 300 # Set to 300 to match the dimension of the 300d GloVe for fair comparison.
    ```

## Experimental Results

### 1. Deep Averaging Network (DAN) Performance

| Model | Embed Dim | Dropout | Initialization | Dev Acc |
| :--- | :---: | :---: | :--- | :---: |
| **Best Model (3-Layer)** | **300** | **0.5** | **GloVe_300d** | **82.6%** |
| Higher Capacity | 300 | 0.0 | GloVe_300d | 82.1% |
| 2-Layer Variant | 300 | 0.5 | GloVe_300d | 82.0% |
| Baseline | 50 | 0.0 | GloVe_50d | 79.5% |
| Random Initialization | 300 | 0.5 | Random | 77.1% |

### 2. Tokenization Comparison (BPE Subword-level vs. Word-level)

| Model | Vocab Size | Initialization | Dev Acc |
| :--- | :---: | :--- | :---: |
| **BPE (Large)** | **10,000** | **Random** | **77.8%** |
| Word-level (Baseline) | ~15,000 | Random | 77.1% |
| BPE (Small) | 2,000 | Random | 74.3% |
