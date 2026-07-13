import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"inititated {device}")

class SimpleCNN(nn.Module):
  def __init__(self, num_classes=10):
    super(SimpleCNN, self).__init__()

    self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride = 1, padding=1)
    self.relu1 = nn.ReLU()
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding =0)
    self.relu2 = nn.ReLU()

    self.fc = nn.Linear(32 * 10 * 10, num_classes)

  def forward(self, x):
    x = self.conv1(x)
    x = self.relu1(x)
    x = self.pool1(x)

    x = self.conv2(x)
    x = self.relu2(x)

    x = x.view(x.size(0), -1)

    x = self.fc(x)
    return x

model = SimpleCNN(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

num_epochs = 2

for epoch in range(num_epochs):
  model.train()
  running_loss = 0.0

  for batch_idx, (images, labels) in enumerate(train_loader):
    images = images.to(device)
    labels = labels.to(device)

    outputs = model(images) # forward pass
    loss = criterion(outputs, labels)

    optimizer.zero_grad() # backward pass
    loss.backward()
    optimizer.step() #update weights(W = W - lr*dW)

    running_loss += loss.item()
    if (batch_idx+1) % 200 == 0:
      print(f"[{epoch+1}/{num_epochs}] | Batch {batch_idx+1} | Loss: {running_loss/200}")
      running_loss = 0.0
