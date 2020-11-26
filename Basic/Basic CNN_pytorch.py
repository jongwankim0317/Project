import torch
from torchvision import datasets, transforms
import torch.functional as F
import torch.nn as nn
import torch.optim as optim
import os
from glob import glob


seed = 1
batch_size = 1024
test_batch_size = 1024
no_cuda = False
use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

torch.manual_seed(seed)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('dataset/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize(mean=(0.5, ), std=(0.5,))
                   ])),
    batch_size=batch_size,
    shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('dataset', train=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,),(0.5))
                   ])),
    batch_size=test_batch_size,
    shuffle=True)

# MODEL

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)

        # print(x.shape) #(for size)
        x = x.view(-1, 4*4*50) # batch size: -1 (don't know)
        x = F.relu(self.fc1(x))
        x = self.fx2(x)
        return F.log_softmax(x, dim=1)

model = Net().to(device)
optimizer = optim.SGD((model.parameters(), lr = 0.001, momentum=0.5)
result = model.forward(image)


