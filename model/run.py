import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import os
from minbpe.basic import BasicTokenizer

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

#here you can load the tokenizer as specified in the repository
@dataclass
class Config:
    n_embed: int = 384
    n_head: int = 8
    n_layer: int = 12
    block_size: int = 256
    dropout: float = 0.2
    batch_size: int = 64
    vocab_size: int = len(tokenizer.vocab)
    head_size: int = 384 // 8

class GPT_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.config = config
        self.embeddings = nn.Embedding(config.vocab_size, config.n_embed)
        self.pos_embeddings = nn.Embedding(config.block_size, config.n_embed)

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.layer_norm = nn.LayerNorm(config.n_embed)
        self.linear_output = nn.Linear(config.n_embed, config.vocab_size)

        self.to(device)

    def forward(self, x, targets):
        embed = self.embeddings(x)
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        pos_embed = self.pos_embeddings(positions)

        x = embed + pos_embed
        for layer in self.blocks:
            x = layer(x)

        x = self.layer_norm(x)
        logits = self.linear_output(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, input_tokens, max_new_tokens, temperature):
        self.eval()
        for _ in range(max_new_tokens):
            cropped = input_tokens[:, -self.config.block_size:]
            output, _ = self.forward(cropped, None)
            logits = output[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
            input_tokens = torch.cat((input_tokens, next_token), dim=1)
        return input_tokens


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.mlp = MLP(config)
        self.layer_norm_1 = nn.LayerNorm(config.n_embed)
        self.layer_norm_2 = nn.LayerNorm(config.n_embed)

    def forward(self, x):
        x = x + self.attention(self.layer_norm_1(x))
        x = x + self.mlp(self.layer_norm_2(x))
        return x

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.l1 = nn.Linear(config.n_embed, config.n_embed * 4)
        self.a1 = nn.ReLU()
        self.l2 = nn.Linear(config.n_embed * 4, config.n_embed)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.l1(x)
        x = self.a1(x)
        x = self.l2(x)
        x = self.dropout(x)
        return x

class Head(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.queries = nn.Linear(config.n_embed, config.head_size)
        self.keys = nn.Linear(config.n_embed, config.head_size)
        self.values = nn.Linear(config.n_embed, config.head_size)
        self.dropout = nn.Dropout(config.dropout)

        self.register_buffer("tril", torch.tril(torch.ones(config.block_size, config.block_size)).to(device))

    def forward(self, x):
        B, T, C = x.shape

        q = self.queries(x)
        k = self.keys(x)
        v = self.values(x)

        weights = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        out = weights @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([Head(config) for _ in range(config.n_head)])
        self.projection = nn.Linear(config.n_head * config.head_size, config.n_embed)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.projection(out)
        out = self.dropout(out)
        return out
    
#here you can run the model with the instructions
