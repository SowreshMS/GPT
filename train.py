import torch
from datasets import load_dataset
import tiktoken
from gpt import GPT

# enc = tiktoken.get_encoding("gpt2")
enc = tiktoken.encoding_for_model("gpt-4")

def process(example):
    ids = enc.encode_ordinary(example['text'])  
    ids.append(enc.eot_token)  
    out = {'ids': ids, 'len': len(ids)}
    return out['ids']

dataset = load_dataset('openwebtext', split='train', streaming=True)
num_epochs = 5
loss_fn = torch.nn.CrossEntropyLoss()
model = GPT(num_layers=12, embed_size=768, num_heads=16, dropout=0, vocab_size=50257, seq_length=1024).cuda()

optimizer = torch.optim.Adam(model.parameters())

for epoch in range(num_epochs):
  losses = []
  for i, batch in enumerate(dataset):
    targets = torch.tensor(process(batch)).cuda()
    targets = F.pad(targets, (0, 1024 - targets.shape[-1]))
    targets = targets.unsqueeze(0)
    logits = model(targets)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    loss /= 32
    loss.backward()

    if (i + 1) % 32 == 0:
      optimizer.step()
      optimizer.zero_grad()

    if (i + 1) % 1000 == 0:
      torch.save(model.state_dict(), '/content')
      print(sum(losses) / len(losses), i)