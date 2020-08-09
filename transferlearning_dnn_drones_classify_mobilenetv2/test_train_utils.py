import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim

train_losses = []
train_acc = []

def train(model, device, trainloader, epoch):
    running_loss = 0.00
    correct = 0.0
    processed = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    pbar = tqdm(trainloader)
    for i, data in enumerate(pbar):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        train_losses.append(loss)
        loss.backward()
        optimizer.step()
        output = outputs.argmax(dim=1, keepdim=True)
        correct = correct + output.eq(labels.view_as(output)).sum().item()
        processed = processed + len(inputs)
        pbar.set_description(desc= f'Loss={loss.item()} Accuracy={100*correct/processed:.2f}')
        train_acc.append(100*correct/processed)

import torch
import torch.nn.functional as F

test_losses = []
test_acc = []

def test(model, device, testloader):
    correct = 0
    total = 0
    test_loss = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_loss += F.nll_loss(outputs, labels, reduction='sum').item()
            #criterion = nn.CrossEntropyLoss()
            #test_loss += criterion(outputs, labels, reduction='sum').item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_losses.append(test_loss)
    print('\nTest Set: Average loss: {}, Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct, len(testloader.dataset),
                                                                        100. * correct / len(testloader.dataset)))
