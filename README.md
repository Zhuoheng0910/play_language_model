# Play Language Model

## Overview

This repository contains codebase for developing a language model using RNN, LSTM and Transformer respectively. You can train your GPT based on your dataset and try to generate text from your trained models.

## Usage

- Train your language models

```bash
python3 train.py --epochs 40 --num_layers 4 --emb_dim 256
```

- Test the performance of your models

```bash
python3 test.py
```

- Generate texts from your prompt tokens with selected model

```bash
python3 generate_text.py --prompt "your prompt" --max_len 60 --temperature 2.0
```

## Dataset

The training and validation dataset in the codebase is Penn Treebank dataset,  a standard corpus for language modeling.  It contains approximately 929,000 training tokens from Wall Street Journal articles. The datasets for test include WikiText-103, WikiText-2 and Tiny Shakespeare Text. You can also replace those data with your desired dataset to train and test.
