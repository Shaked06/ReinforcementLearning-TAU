import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from mnist_utils import *

# Hyper Parameters
input_size = 784
num_classes = 10
num_epochs = 20
batch_size = 100
learning_rate = 1e-3
momentum = 0.9


# MNIST Dataset
train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# Neural Network Model
class Net(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        return out


net = Net(input_size, num_classes)


# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

# Train the Model
loss_vals = []
batch_ind = []
for epoch in range(num_epochs):
    total_epoch_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        # Convert torch tensor to Variable
        images = Variable(images.view(-1, 28*28))
        labels = Variable(labels)

        # Forward
        pred = net(images)
        # Backwad + Optimization
        loss = criterion(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        curr_loss, current = loss.item(), (i + 1) * len(images)
        total_epoch_loss += curr_loss
        if i % 100 == 0:
            print(
                f"loss: {curr_loss:>7f}  [{current:>5d}/{len(train_dataset):>5d}]")

    loss_vals.append(total_epoch_loss / len(train_loader))

# plot the loss curve
loss_plot(values=loss_vals,
          model_name="Logistic Regression Model",
          optimizer="SGD",
          learning_rate=learning_rate,
          batch_size=f"{batch_size}",
          num_epochs=num_epochs,
          file_name="loss-curve-part-1")


# Test the Model
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images.view(-1, 28*28))
    # TODO: implement evaluation code - report accuracy
    pred = net.forward(images)
    correct += (pred.argmax(1) == labels).type(torch.float).sum().item()
    total += labels.size(0)

print('Accuracy of the network on the 10000 test images: %d %%' %
      (100 * correct / total))

# Save the Model
torch.save(net.state_dict(), 'model.pkl')
