# translate.py

import torch
import torch.nn as nn
import pickle
import sys

MAX_LEN = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Load Model and Vocab -----
with open("vocab_en.pkl", "rb") as f: vocab_en = pickle.load(f)
with open("vocab_es.pkl", "rb") as f: vocab_es = pickle.load(f)
with open("inv_es.pkl", "rb") as f: inv_es = pickle.load(f)

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

model = TransformerTranslator(len(vocab_en), len(vocab_es), 100, 2, 2, vocab_en['<pad>']).to(DEVICE)
model.load_state_dict(torch.load("transformer_model.pth", map_location=DEVICE))
model.eval()

def encode(sentence, vocab):
    return [vocab.get(word, vocab['<unk>']) for word in sentence.lower().split()]

def pad_seq(seq, max_len, pad_idx):
    return seq[:max_len] + [pad_idx] * (max_len - len(seq))

def greedy_decode(src_tensor):
    src_tensor = src_tensor.unsqueeze(0).to(DEVICE)
    tgt_indexes = [vocab_es['<sos>']]
    for _ in range(MAX_LEN + 2):
        tgt_tensor = torch.tensor(tgt_indexes).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(src_tensor, tgt_tensor)
        next_token = output[0, -1].argmax(-1).item()
        if next_token == vocab_es['<eos>']:
            break
        tgt_indexes.append(next_token)
    return [inv_es[idx] for idx in tgt_indexes[1:]]

def translate(sentence):
    src_seq = pad_seq(encode(sentence, vocab_en), MAX_LEN, vocab_en['<pad>'])
    src_tensor = torch.tensor(src_seq)
    translation = greedy_decode(src_tensor)
    return ' '.join(translation)

if __name__ == "__main__":
    print("Type an English sentence to translate (or 'quit' to exit):")
    while True:
        inp = input("> ")
        if inp.strip().lower() == 'quit':
            break
        print("→", translate(inp))
