import torch
import torch.nn as nn
import model
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--max_len', type=int, default=50)
parser.add_argument('--prompt', type=str, default='')
parser.add_argument('--temperature', type=float, default=1.0)
args = parser.parse_args()

def load_vocab(vocab_path):
    idx2word = []
    word2idx = {}  # {word:idx}
    with open(vocab_path, encoding='utf-8') as file:
        for idx, word in enumerate(file):
            word = word.strip()
            idx2word.append(word)
            word2idx[word] = idx
    return idx2word, word2idx


def sample(model, word2idx, idx2word, prompt, max_len=50, temperature=1.0, device='cpu'):
    if prompt.strip() == '':
        input_words = ['<eos>']
    else:
        input_words = prompt.strip().split()
    input_ids=[]
    for w in input_words:
        input_ids.append(word2idx.get(w, word2idx['<unk>']))

    generated = input_ids.copy()
    input_tensor = torch.tensor([[input_ids[0]]], dtype=torch.long, device=device)
    hidden = None

    # compute last hidden
    for id in input_ids:
        input_tensor = torch.tensor([[id]], dtype=torch.long, device=device)
        with torch.no_grad():
            _, hidden = model(input_tensor, hidden)

    last_input = torch.tensor([[input_ids[-1]]], dtype=torch.long, device=device)

    for _ in range(max_len):
        with torch.no_grad():
            output, hidden = model(last_input, hidden)  # output: (seq_len=1, batch=1, nvoc)
            logits = output[-1, 0, :] / temperature
            probs = torch.softmax(logits, dim=-1).cpu()

            # option: eliminate output of special tokens
            for special_token in ['<unk>']:
                probs[word2idx[special_token]] = 0.0

            probs /= probs.sum()
            next_word_id = torch.multinomial(probs, num_samples=1).item()
            # next_word_id = torch.argmax(probs).item()

        generated.append(next_word_id)
        last_input = torch.tensor([[next_word_id]], dtype=torch.long, device=device)
    
    # generate words
    generated_words = []
    for i in generated:
        generated_words.append(idx2word[i])

    return ' '.join(generated_words)

def sample_transformer(model, word2idx, idx2word, prompt, max_len=100, temperature=1.0, device='cpu'):
    if prompt.strip() == '':
        input_words = ['<eos>']
    else:
        input_words = prompt.strip().split()

    input_ids=[]
    for w in input_words:
        input_ids.append(word2idx.get(w, word2idx['<unk>']))
        
    generated = input_ids.copy()

    for _ in range(max_len):
        input_tensor = torch.tensor(generated, dtype=torch.long, device=device).unsqueeze(1)  # (seq_len, 1)
        with torch.no_grad():
            output = model(input_tensor)  # output: (seq_len, batch, nvoc)
            logits = output[-1, 0, :] / temperature
            probs = torch.softmax(logits, dim=-1).cpu()

            # option: eliminate output of special tokens
            for special_token in ['<unk>']:
                probs[word2idx[special_token]] = 0

            probs /= probs.sum()
            next_word_idx = torch.multinomial(probs, num_samples=1).item()
        generated.append(next_word_idx)

    generated_words = []
    for i in generated:
        generated_words.append(idx2word[i])
    return ' '.join(generated_words)

idx2word, word2idx = load_vocab('vocab.txt')
nvoc = len(idx2word)
############################## Select Net ##############################
# net = 'RNN'
# loaded_model = model.LMModel_RNN(nvoc=nvoc, dim=256, hidden_size=256, num_layers=4, dropout=0.5)

# net = 'LSTM'
# loaded_model = model.LMModel_LSTM(nvoc=nvoc, dim=256, hidden_size=256, num_layers=4, dropout=0.5)

net = 'Transformer'
loaded_model = model.LMModel_transformer(nvoc=nvoc, num_layers=4, dim=256, nhead=8)
########################################################################
loaded_model.load_state_dict(torch.load(f'checkpoint/model_{net}.pth'))
loaded_model = loaded_model.to(args.device)
loaded_model.eval()

print(f"Prompt: {args.prompt}")
if net == 'Transformer':
    generated_text = sample_transformer(
        model=loaded_model,
        word2idx=word2idx,
        idx2word=idx2word,
        prompt=args.prompt,
        max_len=args.max_len,
        temperature=args.temperature,
        device=args.device
    )
    print(f"{net} Generated text:\n{generated_text}")
else:
    generated_text = sample(
        model=loaded_model,
        word2idx=word2idx,
        idx2word=idx2word,
        prompt=args.prompt,
        max_len=args.max_len,
        temperature=args.temperature,
        device=args.device
    )
    print(f"{net} Generated text:\n{generated_text}")