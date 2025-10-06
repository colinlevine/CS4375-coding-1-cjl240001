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
import string
from argparse import ArgumentParser
import pickle

unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class RNN(nn.Module):
    # Hyperparameters

    #customize
    hidden_dim = 10
    epochs = 200

    #fixed
    learning_rate = 0.01
    minibatch_size = 16
    train_data_path = "training.json"
    val_data_path = "validation.json"

    def __init__(self, input_dim, h):  # Add relevant parameters
        super(RNN, self).__init__()
        self.h = h
        self.numOfLayer = 20
        self.rnn = nn.RNN(input_dim, h, self.numOfLayer, nonlinearity='tanh')
        self.W = nn.Linear(h, 5)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, inputs):
        # [to fill] obtain hidden layer representation (https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)
        output_seq, _ = self.rnn(inputs)

        # [to fill] obtain output layer representations
        output = self.W(output_seq)

        # [to fill] sum over output
        output = output.sum(dim=0)

        # [to fill] obtain probability dist.
        predicted_vector = self.softmax(output)
        return predicted_vector



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
    train_data_path = args.train_data if args.train_data else RNN.train_data_path
    val_data_path = args.val_data if args.val_data else RNN.val_data_path
    hidden_dim = args.hidden_dim if args.hidden_dim else RNN.hidden_dim
    epochs = args.epochs if args.epochs else RNN.epochs

    print("========== Loading data ==========")
    train_data, valid_data = load_data(train_data_path, val_data_path) # X_data is a list of pairs (document, y); y in {0,1,2,3,4}

    # Think about the type of function that an RNN describes. To apply it, you will need to convert the text data into vector representations.
    # Further, think about where the vectors will come from. There are 3 reasonable choices:
    # 1) Randomly assign the input to vectors and learn better embeddings during training; see the PyTorch documentation for guidance
    # 2) Assign the input to vectors using pretrained word embeddings. We recommend any of {Word2Vec, GloVe, FastText}. Then, you do not train/update these embeddings.
    # 3) You do the same as 2) but you train (this is called fine-tuning) the pretrained embeddings further.
    # Option 3 will be the most time consuming, so we do not recommend starting with this

    print("========== Vectorizing data ==========")
    model = RNN(50, hidden_dim)  # Fill in parameters
    model.epochs = epochs
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=model.learning_rate)
    word_embedding = pickle.load(open('./word_embedding.pkl', 'rb'))

    # Initialize lists to store epoch-wise metrics
    training_losses = []
    training_accuracies = []
    validation_losses = []
    validation_accuracies = []

    # Early stopping parameters
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 3

    epoch = 0

    print("========== Training for {} max epochs ==========".format(model.epochs))
    for epoch in range(model.epochs):
        random.shuffle(train_data)
        model.train()
        # You will need further code to operationalize training, ffnn.py may be helpful
        print("Training started for epoch {}".format(epoch + 1))
        train_data = train_data
        correct = 0
        total = 0
        minibatch_size = model.minibatch_size
        N = len(train_data)

        loss_total = 0
        loss_count = 0
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_words, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                input_words = " ".join(input_words)

                # Remove punctuation
                input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()

                # Look up word embedding dictionary
                vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i in input_words ]

                # Transform the input into required shape
                vectors = torch.tensor(np.array(vectors)).view(len(vectors), 1, -1)
                output = model(vectors)

                # Get loss
                example_loss = model.compute_Loss(output.view(1,-1), torch.tensor([gold_label]))

                # Get predicted label
                predicted_label = torch.argmax(output)

                correct += int(predicted_label == gold_label)
                # print(predicted_label, gold_label)
                total += 1
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss

            loss = loss / minibatch_size
            loss_total += loss.data
            loss_count += 1
            loss.backward()
            optimizer.step()
        avg_train_loss = loss_total / loss_count
        train_acc = correct / total
        training_losses.append(avg_train_loss.item())
        training_accuracies.append(train_acc)
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {}".format(epoch + 1, train_acc))
        print("Training loss for epoch {}: {}".format(epoch + 1, avg_train_loss))


        model.eval()
        correct = 0
        total = 0
        val_loss_total = 0
        random.shuffle(valid_data)
        print("Validation started for epoch {}".format(epoch + 1))
        valid_data = valid_data

        for input_words, gold_label in tqdm(valid_data):
            input_words = " ".join(input_words)
            input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
            vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i
                       in input_words]

            vectors = torch.tensor(np.array(vectors)).view(len(vectors), 1, -1)
            output = model(vectors)
            predicted_label = torch.argmax(output)
            correct += int(predicted_label == gold_label)

            # Calculate validation loss
            example_loss = model.compute_Loss(output.view(1,-1), torch.tensor([gold_label]))
            val_loss_total += example_loss.item()

            total += 1
            # print(predicted_label, gold_label)

        validation_accuracy = correct/total
        avg_val_loss = val_loss_total / total
        validation_losses.append(avg_val_loss)
        validation_accuracies.append(validation_accuracy)

        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, validation_accuracy))
        print("Validation loss for epoch {}: {}".format(epoch + 1, avg_val_loss))

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



    # You may find it beneficial to keep track of training accuracy or training loss;

    # Think about how to update the model and what this entails. Consider ffnn.py and the PyTorch documentation for guidance

    # Create logs directory if it doesn't exist
    os.makedirs('rnn_logs', exist_ok=True)

    # Find the next run number
    run_num = 1
    while os.path.exists(f'rnn_logs/run_{run_num}.csv'):
        run_num += 1

    # Write metrics to CSV
    csv_filename = f'rnn_logs/run_{run_num}.csv'

    # Find the epoch with the best (lowest) validation loss
    best_epoch_idx = validation_losses.index(min(validation_losses))

    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write hyperparameters at the top
        csv_writer.writerow(['Hidden Dim', 'Learning Rate', 'Minibatch Size', 'Epochs'])
        csv_writer.writerow([model.h, model.learning_rate, model.minibatch_size, model.epochs])
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
