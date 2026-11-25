from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

data_dir = Path(__file__).resolve().parent / "data"

train_dataset = datasets.MNIST(root=data_dir,
                               train=True,
                               transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307, ), (0.3081, ))]),
                               download=True)
test_dataset = datasets.MNIST(root=data_dir,
                               train=False,
                               transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307, ), (0.3081, ))]),
                               download=True)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=32,
                          shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=32,
                         shuffle=False)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(784, 512)
        self.linear2 = torch.nn.Linear(512, 256)
        self.linear3 = torch.nn.Linear(256, 128)
        self.linear4 = torch.nn.Linear(128, 64)
        self.linear5 = torch.nn.Linear(64, 10)
        self.activate = torch.nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.activate(self.linear1(x))
        x = self.activate(self.linear2(x))
        x = self.activate(self.linear3(x))
        x = self.activate(self.linear4(x))
        x = self.linear5(x)
        return x

model = Model()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
total_loss = 0
for epoch in range(20):
    for batch_idx, (inputs, target) in enumerate(train_loader):
        y_pred = model(inputs)
        loss = criterion(y_pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * target.size(0)

        if batch_idx % 300 == 299:
            print(epoch, batch_idx, loss.item())

correct = 0
total = 0
with torch.no_grad():
    for (inputs, target) in (test_loader):
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
print('测试集准确率:', 100 * correct / total, '%')
