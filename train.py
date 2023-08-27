import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler, autocast
from datasets import load_dataset
from gpt import GPT

# a more robust way of moving our model between gpu or cpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# streaming dataset from huggingface instead of loading it all at once
dataset = load_dataset('openwebtext', split='train', streaming=True)

num_epochs = 5
loss_fn = torch.nn.CrossEntropyLoss()

# model that we will be training
model = GPT(num_layers=6, embed_size=1280, num_heads=16, dropout=0.1, vocab_size=50257, seq_length=1024).to(device)

optimizer = torch.optim.AdamW(model.parameters())

# the scheduler will vary learning rate to boost model performance
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 5, 2)

torch.cuda.empty_cache()

def get_batch(targ, mini_batch):
  x, y = [], []
  for i in range(mini_batch):
    while True:
      idx = np.random.choice(len(targ[0])-1)
      idy = idx + np.random.choice(len(targ[0]) - 1 - idx)

      if not(idy - idx > 1024 or idy - idx < 1):
        break

    x.append(targ[:, idx:idy])
    y.append(targ[:, idx+1:idy+1])

  return x, y

seen = True

for epoch in range(num_epochs):
  losses = []
  for i, batch in enumerate(dataset):
    scheduler.step(epoch)
    sub_data = batch['text']
    targets = model.tokenize(sub_data).to(device)
    targets = targets.unsqueeze(0)

    # getting a random batch of 10 pieces of text from the data
    x, y = get_batch(targets, 10)

    # looping through each batch
    for x1, y1 in zip(x, y):

      logits, loss = model(x1, y1)

      loss /= 40
      losses.append(loss.item() * 40)

      loss.backward()

      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    if (i + 1) % 4 == 0:
      optimizer.step()
      optimizer.zero_grad()

    if (i + 1) % 10 == 0:
      torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'epoch': epoch, 'batch_num': i, 'last_seen_data': batch}, f"/content/drive/My Drive/models/new_model.pt")
      with open("/content", 'w') as file:
        file.write(f'Epoch: {i} Loss: {sum(losses) / len(losses)}')

    if i % 100 == 0:
      print(sum(losses) / len(losses), i)

  losses = []
