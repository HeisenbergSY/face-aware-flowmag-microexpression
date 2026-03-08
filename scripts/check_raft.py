import torch
ckpt = torch.load('models/raft-things.pth')
print(ckpt.keys())  # should NOT contain "generator" or "state_dict"
