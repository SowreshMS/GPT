import torch
import torch.nn as nn
from torch.nn import functional as F

class MLP(nn.Module):
  def __init__(self, embed_size, dropout):
    super().__init__()
    self.fc_in = nn.Linear(embed_size, 4 * embed_size)
    self.gelu = nn.GELU()
    self.fc_out = nn.Linear(4 * embed_size, embed_size)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    x = self.fc_in(x)
    x = self.gelu(x)
    x = self.fc_out(x)
    x = self.dropout(x)
    return x

class Attention(nn.Module):
  def __init__(self, embed_size, num_heads, dropout):
    super().__init__()

    self.key_matrix = nn.Linear(embed_size, embed_size)
    self.queries_matrix = nn.Linear(embed_size, embed_size)
    self.values_matrix = nn.Linear(embed_size, embed_size)
    self.out = nn.Linear(embed_size, embed_size)

    self.attn_drop = nn.Dropout(dropout)
    self.num_heads = num_heads
    self.embed_size = embed_size
    self.dropout = dropout

  def forward(self, x):
    batch, time, channel = x.size()

    keys = self.key_matrix(x)
    queries = self.queries_matrix(x)
    values = self.values_matrix(x)

    keys = keys.view(batch, time, self.num_heads, channel // self.num_heads).transpose(1, 2)
    queries = queries.view(batch, time, self.num_heads, channel // self.num_heads).transpose(1, 2)
    values = values.view(batch, time, self.num_heads, channel // self.num_heads).transpose(1, 2)

    y = F.scaled_dot_product_attention(queries, keys, values, attn_mask=None, dropout_p=self.dropout, is_causal=True)

    y = y.transpose(1, 2).contiguous().view(batch, time, channel)

    y = self.out(y)

    return y

class Block(nn.Module):
  def __init__(self, embed_size, num_heads, dropout):
    super().__init__()
    self.norm1 = nn.LayerNorm(embed_size)
    self.attn = Attention(embed_size, num_heads, dropout)
    self.norm2 = nn.LayerNorm(embed_size)
    self.ff = MLP(embed_size, dropout)

  def forward(self, x):
    x = x + self.attn(self.norm1(x))
    x = x + self.ff(self.norm2(x))
    return x

class GPT(nn.Module):
  def __init__(self, num_layers, embed_size, num_heads, dropout, vocab_size, seq_length):
    super().__init__()
    self.embed = nn.Embedding(vocab_size, embed_size)
    self.pos_embed = nn.Embedding(seq_length, embed_size)
    self.drop = nn.Dropout(dropout)
    self.decoder = nn.ModuleList([Block(embed_size, num_heads, dropout) for _ in range(num_layers)])
    self.norm_out = nn.LayerNorm(embed_size)
    self.out = nn.Linear(embed_size, vocab_size)

  def forward(self, x):
    batch_size = x.shape[0]
    emb = self.embed(x)
    pos = torch.arange(0, x.shape[1], dtype=torch.long, device='cuda')

    x = emb + self.pos_embed(pos).unsqueeze(0)
    x = self.drop(x)

    for block in self.decoder:
      x = block(x)

    x = self.norm_out(x)
    x = self.out(x)

    return x