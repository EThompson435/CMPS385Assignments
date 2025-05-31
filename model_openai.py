# transformer_translator.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
import random

# ----- Hyperparameters -----
HIDDEN_SIZE = 100
NUM_HEADS = 2
NUM_LAYERS = 2
BATCH_SIZE = 16
EPOCHS = 20
MAX_LEN = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Load and Prepare Dataset -----
data = pd.read_csv("english_spanish_200 (1).csv")
data.dropna(inplace=True)
sentences_en = data['english'].str.lower().tolist()
sentences_es = data['spanish'].str.lower().tolist()

# ----- Tokenization -----
def build_vocab(sentences):
    tokens = [word for sentence in sentences for word in sentence.split()]
    freq = Counter(tokens)
    vocab = {word: idx + 4 for idx, (word, _) in enumerate(freq.most_common())}
    vocab['<pad>'] = 0
    vocab['<sos>'] = 1
    vocab['<eos>'] = 2
    vocab['<unk>'] = 3
    return vocab

def encode(sentence, vocab):
    return [vocab.get(word, vocab['<unk>']) for word in sentence.split()]

def pad_seq(seq, max_len, pad_idx):
    return seq[:max_len] + [pad_idx] * (max_len - len(seq))

vocab_en = build_vocab(sentences_en)
vocab_es = build_vocab(sentences_es)
inv_es = {i: w for w, i in vocab_es.items()}

encoded_data = []
for en, es in zip(sentences_en, sentences_es):
    en_seq = pad_seq(encode(en, vocab_en), MAX_LEN, vocab_en['<pad>'])
    es_seq = pad_seq([vocab_es['<sos>']] + encode(es, vocab_es) + [vocab_es['<eos>']], MAX_LEN + 2, vocab_es['<pad>'])
    encoded_data.append((en_seq, es_seq))

train_data, val_data = train_test_split(encoded_data, test_size=0.1, random_state=42)

class TranslationDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        return torch.tensor(src), torch.tensor(tgt)

train_loader = DataLoader(TranslationDataset(train_data), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TranslationDataset(val_data), batch_size=BATCH_SIZE)

# ----- Transformer Model -----
class TransformerTranslator(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size, num_heads, num_layers, pad_idx):
        super().__init__()
        self.src_embed = nn.Embedding(input_dim, hidden_size, padding_idx=pad_idx)
        self.tgt_embed = nn.Embedding(output_dim, hidden_size, padding_idx=pad_idx)
        self.transformer = nn.Transformer(d_model=hidden_size, nhead=num_heads,
                                          num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers)
        self.fc_out = nn.Linear(hidden_size, output_dim)
        self.pad_idx = pad_idx

    def make_src_mask(self, src):
        return (src == self.pad_idx).transpose(0, 1)

    def forward(self, src, tgt):
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)
        src_mask = self.make_src_mask(src)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(0)).to(DEVICE)
        src_emb = self.src_embed(src)
        tgt_emb = self.tgt_embed(tgt)
        output = self.transformer(src_emb, tgt_emb, src_key_padding_mask=src_mask, tgt_mask=tgt_mask)
        return self.fc_out(output).transpose(0, 1)

model = TransformerTranslator(len(vocab_en), len(vocab_es), HIDDEN_SIZE, NUM_HEADS, NUM_LAYERS, vocab_en['<pad>']).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(ignore_index=vocab_es['<pad>'])

# ----- Training Loop -----
def train(model, loader):
    model.train()
    total_loss = 0
    for src, tgt in loader:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        output = model(src, tgt_input)
        output = output.reshape(-1, output.shape[-1])
        tgt_output = tgt_output.reshape(-1)
        loss = criterion(output, tgt_output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            output = model(src, tgt_input)
            output = output.reshape(-1, output.shape[-1])
            tgt_output = tgt_output.reshape(-1)
            loss = criterion(output, tgt_output)
            total_loss += loss.item()
    return total_loss / len(loader)

# ----- Train Model -----
for epoch in range(EPOCHS):
    train_loss = train(model, train_loader)
    val_loss = evaluate(model, val_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# ----- Save model and vocab -----
torch.save(model.state_dict(), "transformer_model.pth")
import pickle
with open("vocab_en.pkl", "wb") as f: pickle.dump(vocab_en, f)
with open("vocab_es.pkl", "wb") as f: pickle.dump(vocab_es, f)
with open("inv_es.pkl", "wb") as f: pickle.dump(inv_es, f)
print("Model and vocabularies saved.")
