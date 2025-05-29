import math
import torch
import torch.nn as nn

# Language model is composed of three parts: a word embedding layer, a rnn network and a output layer.
# The word embedding layer have input as a sequence of word index (in the vocabulary) and output a sequence of vector where each one is a word embedding.
# The rnn network has input of each word embedding and output a hidden feature corresponding to each word embedding.
# The output layer has input as the hidden feature and output the probability of each word in the vocabulary.

class LMModel_RNN(nn.Module):
    """
    RNN-based language model:
    1) Embedding layer
    2) Vanilla RNN network
    3) Output linear layer
    """
    def __init__(self, nvoc, dim=256, hidden_size=256, num_layers=2, dropout=0.5):
        super(LMModel_RNN, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(nvoc, dim)
        ########################################
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dim = dim

        self.W_i = nn.ParameterList()
        self.W_h = nn.ParameterList()
        self.b_h = nn.ParameterList()

        for i in range(num_layers):
            if i == 0:
                W_i_param = nn.Parameter(torch.Tensor(dim, hidden_size))
            else:
                W_i_param = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
            self.W_i.append(W_i_param)

            W_h_param = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
            self.W_h.append(W_h_param)

            b_h_param = nn.Parameter(torch.Tensor(hidden_size))
            self.b_h.append(b_h_param)
        ########################################
        self.decoder = nn.Linear(hidden_size, nvoc)
        self.init_weights()

    def init_weights(self):
        init_uniform = 0.1
        self.encoder.weight.data.uniform_(-init_uniform, init_uniform)
        for i in range(self.num_layers):
            self.W_i[i].data.uniform_(-init_uniform, init_uniform)
            self.W_h[i].data.uniform_(-init_uniform, init_uniform)
            self.b_h[i].data.zero_()
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)
    
    def forward(self, input, hidden=None):
        # input shape: (seq_len, batch_size)
        embeddings = self.drop(self.encoder(input))  # (seq_len, batch_size, dim)
        ########################################
        seq_len, batch_size, dim = embeddings.size()

        if hidden is None:
            hidden = []
            for _ in range(self.num_layers):
                h = torch.zeros(batch_size, self.hidden_size, device=embeddings.device)  # (batch_size, hidden_size)
                hidden.append(h)

        output = []
        for t in range(seq_len):
            e = embeddings[t]  # (batch_size, dim)
            new_hidden = []
            for l in range(self.num_layers):
                h_prev = hidden[l]  # (batch_size, hidden_size)
                h = torch.sigmoid(torch.matmul(e, self.W_i[l]) + torch.matmul(h_prev, self.W_h[l]) + self.b_h[l])
                new_hidden.append(h)
                e = h
            hidden = new_hidden
            output.append(e.unsqueeze(0))
        output = torch.cat(output, dim=0)  # (seq_len, batch_size, hidden_size)
        ########################################

        output = self.drop(output)
        decoded = self.decoder(output.view(-1, output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(-1)), hidden


class LMModel_LSTM(nn.Module):
    """
    LSTM-based language model:
    (1) Embedding layer
    (2) LSTM network
    (3) Output linear layer
    """
    def __init__(self, nvoc, dim=256, hidden_size=256, num_layers=2, dropout=0.5):
        super(LMModel_LSTM, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(nvoc, dim)
        ########################################
        self.lstm = nn.LSTM(input_size=dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=False)
        ########################################
        self.decoder = nn.Linear(hidden_size, nvoc)
        self.init_weights()

    def init_weights(self):
        init_uniform = 0.1
        self.encoder.weight.data.uniform_(-init_uniform, init_uniform)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)

    def forward(self, input, hidden=None):
        # input shape: (seq_len, batch_size)
        embeddings = self.drop(self.encoder(input))  # (seq_len, batch, dim)

        ########################################
        output, hidden = self.lstm(embeddings, hidden)
        ########################################

        output = self.drop(output)
        decoded = self.decoder(output.view(-1, output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(-1)), hidden


# Official Implementation of Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

   
class LMModel_transformer(nn.Module):
    def __init__(self, nvoc, dim=256, nhead=8, num_layers=4):
        super(LMModel_transformer, self).__init__()
        self.drop = nn.Dropout(0.5)
        self.encoder = nn.Embedding(nvoc, dim)
        ########################################
        self.dim = dim
        self.positional_encoder = PositionalEncoding(dim, dropout=0.1)
        self.encoder_layers = nn.TransformerEncoderLayer(d_model=dim, nhead=nhead,dim_feedforward=dim*4,
                                                        dropout=0.1, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers, num_layers=num_layers)
        ########################################

        self.decoder = nn.Linear(dim, nvoc)
        self.init_weights()

    def init_weights(self):
        init_uniform = 0.1
        self.encoder.weight.data.uniform_(-init_uniform, init_uniform)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)

    def forward(self, input):
        #print(input.device)
        embeddings = self.drop(self.encoder(input))

        ########################################
        seq_len = embeddings.size(0)
        src_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(input.device.type)
        src = embeddings * math.sqrt(self.dim)
        src = self.positional_encoder(src)
        output = self.transformer_encoder(src, mask=src_mask)  # (seq_len, batch_size, dim)
        ########################################
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1))