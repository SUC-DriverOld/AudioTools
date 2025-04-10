import torch

model_1 = torch.load('model_1.ckpt', map_location='cpu')
model_2 = torch.load('model_2.ckpt', map_location='cpu')
model_3 = torch.load('model_3.ckpt', map_location='cpu')

# Combine the models
fused_weights = {}
for key in model_1.keys():
    fused_weights[key] = 0.5 * model_1[key] + 0.25 * model_2[key] + 0.25 * model_3[key]

# Save the fused model
torch.save(fused_weights, 'fused_model.ckpt')