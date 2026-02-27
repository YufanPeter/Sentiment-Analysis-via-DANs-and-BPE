# models.py

from IPython import embed
from altair import Data
from contourpy import dechunk_filled # type: ignore
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
from sentiment_data import read_sentiment_examples
from torch.utils.data import Dataset, DataLoader
import time
import argparse
import matplotlib.pyplot as plt # type: ignore
from torch.utils.data import DataLoader
from BOWmodels import SentimentDatasetBOW, NN2BOW, NN3BOW
from DANmodels import SentimentDatasetDAN, SentimentDatasetBPE, DAN
from sentiment_data import read_word_embeddings
from torch.nn.utils.rnn import pad_sequence
from BPE import BPE
import test

def pad(batch):
    """
    Custom collate function for the DataLoader.
    
    Args:
        batch (list): A list of tuples (input_indices, label) from the Dataset.
                      
    Returns:
        inputs_pad (torch.Tensor): Padded input batch of shape [batch_size, max_seq_len].
        labels_stacked (torch.Tensor): Batch of labels of shape [batch_size].
    """
    inputs = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    inputs_pad = pad_sequence(inputs, batch_first=True, padding_value=0)
    labels_stacked = torch.stack(labels)
    
    return inputs_pad, labels_stacked

# Training function
def train_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(data_loader):
        if isinstance(model, DAN):
            X = X.long()
        else:
            X = X.float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_train_loss = train_loss / num_batches
    accuracy = correct / size
    return accuracy, average_train_loss


# Evaluation function
def eval_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.eval()
    eval_loss = 0
    correct = 0
    for batch, (X, y) in enumerate(data_loader):
        if isinstance(model, DAN):
            X = X.long()
        else:
            X = X.float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        eval_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    average_eval_loss = eval_loss / num_batches
    accuracy = correct / size
    return accuracy, average_eval_loss


# Experiment function to run training and evaluation for multiple epochs
def experiment(model, train_loader, test_loader):
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) # type: ignore

    all_train_accuracy = []
    all_test_accuracy = []

    best_test_accuracy = 0.0
    best_epoch = 0
    
    patience = 10
    no_improve_epochs = 0 

    for epoch in range(100):
        train_accuracy, train_loss = train_epoch(train_loader, model, loss_fn, optimizer)
        all_train_accuracy.append(train_accuracy)

        test_accuracy, test_loss = eval_epoch(test_loader, model, loss_fn, optimizer)
        all_test_accuracy.append(test_accuracy)

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_epoch = epoch + 1
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if epoch % 10 == 9 or no_improve_epochs >= patience:
            print(f'Epoch #{epoch + 1}: train accuracy {train_accuracy:.3f}, dev accuracy {test_accuracy:.3f}')

        if no_improve_epochs >= patience:
            print(f"\nEarly stopping triggered at Epoch #{epoch + 1}!")
            print(f"No improvement for {patience} epochs.")
            break
    
    print(f"\nBest Dev Accuracy is {best_test_accuracy:.3f} at Epoch #{best_epoch}")
    return all_train_accuracy, all_test_accuracy, best_test_accuracy


def main():

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run model training based on specified model type')
    parser.add_argument('--model', type=str, required=True, help='Model type to train (e.g., BOW)')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Load dataset
    start_time = time.time()

    train_data = SentimentDatasetBOW("data/train.txt")
    dev_data = SentimentDatasetBOW("data/dev.txt", vectorizer=train_data.vectorizer, train=False)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(dev_data, batch_size=16, shuffle=False)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Data loaded in : {elapsed_time} seconds")


    # Check if the model type is "BOW"
    if args.model == "BOW":
        # Train and evaluate NN2
        start_time = time.time()
        print('\n2 layers:')
        nn2_train_accuracy, nn2_test_accuracy, best_acc = experiment(NN2BOW(input_size=512, hidden_size=100), train_loader, test_loader)

        # Train and evaluate NN3
        print('\n3 layers:')
        nn3_train_accuracy, nn3_test_accuracy, best_acc = experiment(NN3BOW(input_size=512, hidden_size=100), train_loader, test_loader)

        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_train_accuracy, label='2 layers')
        plt.plot(nn3_train_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy for 2, 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the training accuracy figure
        training_accuracy_file = 'train_accuracy.png'
        plt.savefig(training_accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot the testing accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_test_accuracy, label='2 layers')
        plt.plot(nn3_test_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Dev Accuracy')
        plt.title('Dev Accuracy for 2 and 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the testing accuracy figure
        testing_accuracy_file = 'dev_accuracy.png'
        plt.savefig(testing_accuracy_file)
        print(f"Dev accuracy plot saved as {testing_accuracy_file}\n\n")

        # plt.show()

    elif args.model == "DAN":
        embeddings_file = 'data/glove.6B.50d-relativized.txt'
        # embeddings_file = 'data/glove.6B.300d-relativized.txt'
        print(f"Currently loading embeddings from {embeddings_file}")
        word_embeddings = read_word_embeddings(embeddings_file)
        vocab_size = len(word_embeddings.word_indexer)
        print(f"Vocab size: {vocab_size}")
        
        # Switch between learning embeddings from scratch (Random) vs. using GloVe (Pre-trained)
        USE_RANDOM = False
        if USE_RANDOM:
            vocab_size = len(word_embeddings.word_indexer)
            embed_dim = word_embeddings.get_embedding_length()
            embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        else:
            embedding_layer = word_embeddings.get_initialized_embedding_layer(frozen=False)

        train_data = SentimentDatasetDAN("data/train.txt", word_embeddings)
        dev_data = SentimentDatasetDAN("data/dev.txt", word_embeddings)
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=pad)
        test_loader = DataLoader(dev_data, batch_size=64, shuffle=False, collate_fn=pad)

        dan_model = DAN(embedding_layer, hidden_dim=100)
        DAN_train_acc, DAN_dev_acc, best_acc = experiment(dan_model, train_loader, test_loader)

        plt.figure(figsize=(8, 6))
        plt.plot(DAN_train_acc, label='Train Accuracy')
        plt.plot(DAN_dev_acc, label='Dev Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('DAN Training Accuracy (GloVe 50d)')
        plt.legend()
        plt.grid()

        plot_file = 'DAN_50d_accuracy.png'
        plt.savefig(plot_file)
        print(f"\nPlot saved to {plot_file}")
    
    elif args.model == "BPE":
        vocab_size = 2000
        print(f"Currently loading embeddings from BPE with vocab size {vocab_size}")

        bpe = BPE(vocab_size=vocab_size)
        bpe.fit("data/train.txt")

        current_vocab = len(bpe.indexer)
        embed_dim = 300
        embedding_layer = nn.Embedding(current_vocab, embed_dim)

        train_data = SentimentDatasetBPE("data/train.txt", bpe)
        dev_data = SentimentDatasetBPE("data/dev.txt", bpe)
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=pad)
        test_loader = DataLoader(dev_data, batch_size=64, shuffle=False, collate_fn=pad)
        
        bpe_model = DAN(embedding_layer, hidden_dim=100)
        bpe_train_acc, bpe_dev_acc, best_acc = experiment(bpe_model, train_loader, test_loader)

        plt.figure(figsize=(8, 6))
        plt.plot(bpe_train_acc, label='Train Accuracy')
        plt.plot(bpe_dev_acc, label='Dev Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('BPE Training Accuracy')
        plt.legend()
        plt.grid()

        plot_file = 'BPE_accuracy.png'
        plt.savefig(plot_file)
        print(f"\nPlot saved to {plot_file}")

if __name__ == "__main__":
    main()
