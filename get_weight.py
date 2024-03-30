# 简化权重
import torch

model = torch.load('./weights/epoch48.pt')

model['ema'] = None
model['updates'] = None
model['optimizer'] = None

torch.save(model, 'test.pt')