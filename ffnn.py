import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
import json
import csv
from argparse import ArgumentParser


unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class FFNN(nn.Module):
    # Hyperparameters

    #customize
    hidden_dim = 10
    epochs = 50

    #fixed
    learning_rate = 0.01
    momentum = 0.9
    minibatch_size = 16
    train_data_path = "training.json"
    val_data_path = "validation.json"

    def __init__(self, input_dim, h=None):
        super(FFNN, self).__init__()
        self.h = h if h is not None else self.hidden_dim
        self.W1 = nn.Linear(input_dim, self.h)
        self.activation = nn.ReLU() # The rectified linear unit; one valid choice of activation function
        self.output_dim = 5
        self.W2 = nn.Linear(self.h, self.output_dim)

        self.softmax = nn.LogSoftmax() # The softmax function that converts vectors into probability distributions; computes log probabilities for computational benefits
        self.loss = nn.NLLLoss() # The cross-entropy/negative log likelihood loss taught in class

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_vector):
        # [to fill] obtain first hidden layer representation
        hidden_layer = self.W1(input_vector)
        hl_act = self.activation(hidden_layer)

        # [to fill] obtain output layer representation
        output = self.W2(hl_act)
        # [to fill] obtain probability dist.
        out_act = self.softmax(output)

        return out_act


# Returns: 
# vocab = A set of strings corresponding to the vocabulary
def make_vocab(data):
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab 


# Returns:
# vocab = A set of strings corresponding to the vocabulary including <UNK>
# word2index = A dictionary mapping word/token to its index (a number in 0, ..., V - 1)
# index2word = A dictionary inverting the mapping of word2index
def make_indices(vocab):
    vocab_list = sorted(vocab)
    vocab_list.append(unk)
    word2index = {}
    index2word = {}
    for index, word in enumerate(vocab_list):
        word2index[word] = index 
        index2word[index] = word 
    vocab.add(unk)
    return vocab, word2index, index2word 


# Returns:
# vectorized_data = A list of pairs (vector representation of input, y)
def convert_to_vector_representation(data, word2index):
    vectorized_data = []
    for document, y in data:
        vector = torch.zeros(len(word2index)) 
        for word in document:
            index = word2index.get(word, word2index[unk])
            vector[index] += 1
        vectorized_data.append((vector, y))
    return vectorized_data



def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = []
    val = []
    for elt in training:
        tra.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in validation:
        val.append((elt["text"].split(),int(elt["stars"]-1)))

    return tra, val


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, default=None, help = "hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, default=None, help = "num of epochs to train")
    parser.add_argument("--train_data", default=None, help = "path to training data")
    parser.add_argument("--val_data", default=None, help = "path to validation data")
    parser.add_argument("--test_data", default = "to fill", help = "path to test data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    # fix random seeds
    random.seed(42)
    torch.manual_seed(42)

    # Use class variables as defaults if CLI args not provided
    train_data_path = args.train_data if args.train_data else FFNN.train_data_path
    val_data_path = args.val_data if args.val_data else FFNN.val_data_path
    hidden_dim = args.hidden_dim if args.hidden_dim else FFNN.hidden_dim
    epochs = args.epochs if args.epochs else FFNN.epochs

    # load data
    print("========== Loading data ==========")
    train_data, valid_data = load_data(train_data_path, val_data_path) # X_data is a list of pairs (document, y); y in {0,1,2,3,4}
    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)

    print("========== Vectorizing data ==========")
    train_data = convert_to_vector_representation(train_data, word2index)
    valid_data = convert_to_vector_representation(valid_data, word2index)


    model = FFNN(input_dim = len(vocab), h = hidden_dim)
    model.epochs = epochs
    optimizer = optim.SGD(model.parameters(),lr=model.learning_rate, momentum=model.momentum)

    # Initialize lists to store epoch-wise metrics
    training_losses = []
    training_accuracies = []
    validation_losses = []
    validation_accuracies = []

    # Early stopping parameters
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 3

    print("========== Training for {} max epochs ==========".format(model.epochs))
    for epoch in range(model.epochs):
        model.train()
        optimizer.zero_grad()
        loss = None
        correct = 0
        total = 0
        total_train_loss = 0
        start_time = time.time()
        print("Training started for epoch {}".format(epoch + 1))
        random.shuffle(train_data) # Good practice to shuffle order of training data
        minibatch_size = model.minibatch_size
        N = len(train_data) 
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
            loss = loss / minibatch_size
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
        print("Training completed for epoch {}".format(epoch + 1))
        train_acc = correct / total
        avg_train_loss = total_train_loss / (N // minibatch_size)
        training_losses.append(avg_train_loss)
        training_accuracies.append(train_acc)
        print("Training accuracy for epoch {}: {}".format(epoch + 1, train_acc))
        print("Training loss for epoch {}: {}".format(epoch + 1, avg_train_loss))
        print("Training time for this epoch: {}".format(time.time() - start_time))


        loss = None
        correct = 0
        total = 0
        total_val_loss = 0
        start_time = time.time()
        print("Validation started for epoch {}".format(epoch + 1))
        minibatch_size = model.minibatch_size
        N = len(valid_data) 
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = valid_data[minibatch_index * minibatch_size + example_index]
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
            loss = loss / minibatch_size
            total_val_loss += loss.item()
        print("Validation completed for epoch {}".format(epoch + 1))
        val_acc = correct / total
        avg_val_loss = total_val_loss / (N // minibatch_size)
        validation_losses.append(avg_val_loss)
        validation_accuracies.append(val_acc)
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, val_acc))
        print("Validation loss for epoch {}: {}".format(epoch + 1, avg_val_loss))
        print("Validation time for this epoch: {}".format(time.time() - start_time))

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            print("Validation loss improved!")
        else:
            patience_counter += 1
            print(f"No improvement. Patience counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

    # log the training and validation losses in a neat csv
    # should check what the last run number was and create a new file each time

    # Create logs directory if it doesn't exist
    os.makedirs('ffnn_logs', exist_ok=True)

    # Find the next run number
    run_num = 1
    while os.path.exists(f'ffnn_logs/run_{run_num}.csv'):
        run_num += 1

    # Write metrics to CSV
    csv_filename = f'ffnn_logs/run_{run_num}.csv'

    # Find the epoch with the best (lowest) validation loss
    best_epoch_idx = validation_losses.index(min(validation_losses))

    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write hyperparameters at the top
        csv_writer.writerow(['Hidden Dim', 'Learning Rate', 'Momentum', 'Minibatch Size', 'Epochs'])
        csv_writer.writerow([model.h, model.learning_rate, model.momentum, model.minibatch_size, model.epochs])
        csv_writer.writerow(['Best Val Loss', min(validation_losses)])
        csv_writer.writerow([])  # Empty row for separation
        # Write epoch metrics
        csv_writer.writerow(['Epoch', 'Training Loss', 'Training Accuracy', 'Validation Loss', 'Validation Accuracy', 'Best'])
        for i in range(len(training_losses)):
            best_marker = '*' if i == best_epoch_idx else ''
            csv_writer.writerow([
                i + 1,
                training_losses[i],
                training_accuracies[i],
                validation_losses[i],
                validation_accuracies[i],
                best_marker
            ])

    print(f"Training metrics saved to {csv_filename}")
