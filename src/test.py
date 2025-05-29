import torch
import torch.nn as nn
import math
import os
import data
import model

class Args:
    def __init__(self):
        self.test_batch_size = 10
        self.max_sql = 256
        self.emb_dim = 256
        self.hidden_size = 256
        self.num_layers = 4
        self.num_heads = 8
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = Args()

def load_vocab(vocab_path="vocab.txt"):
    old_vocab = {}
    with open(vocab_path, "r") as f:
        for idx, word in enumerate(f.read().splitlines()):
            old_vocab[word] = idx
    return old_vocab
    
class StrictCorpus(data.Corpus):
    def __init__(self, path, batch_size, max_sql, old_vocab):
        self.old_vocab = old_vocab  # {word: id}
        super().__init__(path, batch_size, max_sql)
        self.vocabulary = list(old_vocab.keys())
        self.word_id = old_vocab.copy()

    def tokenize(self, file_name):
        file_lines = open(file_name, 'r').readlines()
        tokens = []
        
        for line in file_lines:
            words = line.strip().split() + ['<eos>']
            for word in words:
                if word in self.old_vocab:
                    tokens.append(self.old_vocab[word])
            
        return torch.LongTensor(tokens)


old_vocab = load_vocab()
print(f"Old vocab size: {len(old_vocab)}")

test_data_loader = StrictCorpus(
    path="../data/",
    batch_size={'train': 10, 'valid': args.test_batch_size},
    max_sql=args.max_sql,
    old_vocab=old_vocab
)

############################## Select Net ##############################
# net = 'RNN'
# loaded_model = model.LMModel_RNN(nvoc=len(old_vocab), dim=args.emb_dim, hidden_size=args.hidden_size, num_layers=args.num_layers)

# net = 'LSTM'
# loaded_model = model.LMModel_LSTM(nvoc=len(old_vocab), dim=args.emb_dim, hidden_size=args.hidden_size, num_layers=args.num_layers)

net = 'Transformer'
loaded_model = model.LMModel_transformer(nvoc=len(old_vocab), dim=args.emb_dim, nhead=args.num_heads, num_layers=args.num_layers)
########################################################################
loaded_model.load_state_dict(torch.load(f"checkpoint/model_{net}.pth"))
loaded_model = loaded_model.to(args.device)
loaded_model.eval()

criterion = nn.CrossEntropyLoss()
test_data_loader.set_valid()
test_data, target, end_flag = test_data_loader.get_batch()
idx = 0
avg_loss = 0
print("Testing")

while not end_flag:
    with torch.no_grad():
        test_data, target, end_flag = test_data_loader.get_batch()
        test_data = test_data.to(args.device)
        target = target.to(args.device)
        decode = loaded_model(test_data)
        # decode, _ = loaded_model(test_data)
        
        loss = criterion(decode.view(decode.size(0) * decode.size(1), -1), target.view(-1))
        avg_loss += loss.item()
        idx += 1

avg_loss = avg_loss / idx
perplexity = math.exp(avg_loss)

print(f"{net} Average Loss: {avg_loss:.2f}")
print(f"{net} Test Perplexity: {perplexity:.2f}")