import torch
import torch.nn as nn

sample_batch = torch.randn(64, 1, 28, 28)

conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)

print("Input shape:", sample_batch.shape)

out = conv1(sample_batch)
print("After Cnv1(3x3, pad=1):", out.shape)

out = pool1(out)
print("After pool1(2x2, stride=2):", out.shape)

out = conv2(out)
print("After conv2(5x5, pad=0):", out.shape)

flattened = out.view(out.size(0), -1)
print("Flattened shape for linear layer:", flattened.shape)
