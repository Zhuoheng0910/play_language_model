# coding: utf-8
import argparse
import time
import math
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn

import data
import model
import os
import os.path as osp

parser = argparse.ArgumentParser(description='PyTorch ptb Language Model')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--train_batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=10, metavar='N',
                    help='eval batch size')
parser.add_argument('--max_sql', type=int, default=256,
                    help='sequence length')
parser.add_argument('--seed', type=int, default=1234,
                    help='set random seed')
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--emb_dim', type=int, default=256)
parser.add_argument('--use_pe', action="store_true")
parser.add_argument('--cuda', action='store_true', help='use CUDA device')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU device id used')

args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

# Use gpu or cpu to train
use_gpu = True

if use_gpu:
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(args.gpu_id)
else:
    device = torch.device("cpu")

# load data
train_batch_size = args.train_batch_size
eval_batch_size = args.eval_batch_size
batch_size = {'train': train_batch_size,'valid':eval_batch_size}
data_loader = data.Corpus("../data/", batch_size, args.max_sql)

# Save the vocab file
# with open("vocab.txt", "w") as vocab_file:
#     for word in data_loader.vocabulary:
#         vocab_file.write(word + "\n")

########################################
net = 'RNN'
model = model.LMModel_RNN(nvoc=len(data_loader.vocabulary), dim=args.emb_dim, hidden_size=256, num_layers=args.num_layers)

# net = 'LSTM'
# model = model.LMModel_LSTM(nvoc=len(data_loader.vocabulary), dim=args.emb_dim, hidden_size=256, num_layers=args.num_layers)

# net = 'Transformer'
# model = model.LMModel_transformer(nvoc=len(data_loader.vocabulary), dim=args.emb_dim, nhead=args.num_heads, num_layers=args.num_layers)
########################################################################

lr = 1e-3
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr)
criterion = nn.CrossEntropyLoss()

train_losses = []
valid_losses = []

# Evaluation Function
# Calculate the average cross-entropy loss between the prediction and the ground truth word.
# And then exp(average cross-entropy loss) is perplexity.

def evaluate():
    data_loader.set_valid()
    data, target, end_flag = data_loader.get_batch()
    model.eval()
    idx = 0
    avg_loss = 0
    print(f"Validating")
    while not end_flag:
        with torch.no_grad():
            data, target, end_flag = data_loader.get_batch()
            data = data.to(device)
            target = target.to(device)
            decode, _ = model(data)
            # decode = model(data)

            # Calculate cross-entropy loss
            loss = criterion(decode.view(decode.size(0) * decode.size(1), -1), target)
            avg_loss += loss
            idx += 1
    print(f"The average loss is {avg_loss / idx}")
    return avg_loss.item() / idx


# Train Function
def train():
    data_loader.set_train()
    data, target, end_flag = data_loader.get_batch()
    model.train()
    idx = 0
    avg_loss = 0
    while not end_flag:
        data, target, end_flag = data_loader.get_batch()
        data = data.to(device)
        target = target.to(device)
        decode, _ = model(data)
        # decode = model(data)

        # Calculate cross-entropy loss
        optimizer.zero_grad()
        loss = criterion(decode.view(decode.size(0)*decode.size(1), -1), target)
        loss.backward()
        optimizer.step()
        if (idx+1) % 50 == 0:
            print(f"The loss is {loss}")
        idx += 1
        avg_loss += loss
    return avg_loss.item() / idx


# Loop over epochs.
train_perplexity = []
valid_perplexity = []
for epoch in range(1, args.epochs+1):
    print(f"Start training epoch {epoch}")
    train_loss = train()
    valid_loss = evaluate()
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    train_perplexity.append(math.exp(train_loss))
    valid_perplexity.append(math.exp(valid_loss))

torch.save(model.state_dict(), f'checkpoint/model_{net}.pth')
print(f"Model saved to 'checkpoint/model_{net}.pth'.")

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, args.epochs + 1), train_losses, label='Training Loss', marker='o')
plt.plot(range(1, args.epochs + 1), valid_losses, label='Validation Loss', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title(f'{net} Loss')
plt.legend()
plt.grid()
plt.savefig(f'loss_curve/{net}.png')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(range(1, args.epochs + 1), train_perplexity, label='Training Perplexcity', marker='o')
plt.plot(range(1, args.epochs + 1), valid_perplexity, label='Validation Perplexcity', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Perplexcity')
plt.title(f'{net} Perplexcity')
plt.legend()
plt.grid()
plt.savefig(f'ppl_curve/{net}.png')
plt.show()

print(f"Train Perpelexity: {train_perplexity}")
print(f"Valid Perpelexity: {valid_perplexity}")

