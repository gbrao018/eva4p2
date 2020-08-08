from __future__ import print_function
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import copy
from torchsummary import summary
from torchvision import datasets, transforms
import numpy as np

def train_model(model, device, train_loader, optimizer,scheduler, epoch, train_losses, train_acc, criteria, store_mode ='epoch', doL1 = 0, doL2 = 0,LAMBDA = 0):
  print('L1=',doL1,';L2=',doL2,';LAMBDA=',LAMBDA, 'epoch=',epoch)
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  train_loss = 0
    
  #as batches
  #data is the touple returned from get_item
  for batch_idx, (inputs, targets) in enumerate(pbar):
    # get samples
    #data, target = data[0].to(device), target.to(device)
    #print(data_touple)
    #inputs, targets = data_touple
    inputs=inputs.to(device)
    targets=targets.to(device)
    #print('inputs=',inputs.shape,'targets=',targets.shape)
    
    #print('data=',len(data),';target=',len(target))

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(inputs)
    

    # Calculate loss
    #print('y_pred=',len(y_pred.dataset),'target=',len(target.dataset))
    #loss = F.nll_loss(y_pred, target)
    #criteria = nn.CrossEntropyLoss()
    loss = criteria(y_pred, targets) 
    reg_loss=0
    if (doL1 == 1):
      for p in model.parameters():  
        reg_loss += torch.sum(torch.abs(p.data))
    if (doL2 == 1):
      for p in model.parameters():
        reg_loss += torch.sum(p.data.pow(2))    
    
    loss+=LAMBDA*reg_loss
            
    train_losses.append(loss)

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Update pbar-tqdm
    #after batch, scheduler.step
    scheduler.step()    
    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(targets.view_as(pred)).sum().item()
    processed += len(inputs)
    train_loss += loss.item()
    """
    if store_mode == 'epoch':   # Store loss and accuracy
        accuracy = 100 * correct / processed
        if not train_losses is None:
            train_losses.append(loss.item())
        if not accuracies is None:
            train_acc.append(accuracy)  
    
    elif  store_mode == 'mini_batch':  # Store loss and accuracy
        batch_accuracy = 100 * correct / processed
        if not train_losses is None:
            train_losses.append(loss.item())
        if not train_acc is None:
            train_acc.append(batch_accuracy)
    """
  train_loss /= len(train_loader.dataset) 
  train_losses.append(train_loss) 
  train_accuracy = 100. * correct / len(train_loader.dataset)
  train_acc.append(train_accuracy)
  print('\n epoch=',epoch,'\nTrain set: Average loss:',train_loss,'Accuracy:',train_accuracy)
    
  #pbar.set_description(desc=  epoch={epoch} f'AverageLoss={loss.item()} Accuracy={100*correct/processed:0.2f}')     
      #pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
  #train_acc.append(100*correct/processed)
    

def test_model(model, device, test_loader,test_losses,test_acc,criteria, correct_samples, incorrect_samples, sample_count=30, last_epoch=False):
    model.eval()
    test_loss = 0
    correct = 0
    #criteria = nn.CrossEntropyLoss()
            
    with torch.no_grad():
        for (inputs, targets) in test_loader:
            #img_batch = data
            #print(targets)
            img_batch = inputs
            #data, target = data.to(device), target.to(device)
            #inputs, targets = data_touple
            inputs=inputs.to(device)
            targets=targets.to(device)
            #print('data=',len(data),';target=',len(target))
            output = model(inputs)
            #test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            #test_loss += criteria(output, target).item()
            test_loss += criteria(output, targets).item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            result = pred.eq(targets.view_as(pred))
            """
            if last_epoch:
                #print('last_epoch=',last_epoch)
                for i in range(len(list(result))):
                    if not list(result)[i] and len(incorrect_samples) < sample_count:
                        incorrect_samples.append({
                            'prediction': list(pred)[i],
                            'label': list(targets.view_as(pred))[i],
                            'image': img_batch[i]
                            
                        })
                    elif list(result)[i] and len(correct_samples) < sample_count:
                        correct_samples.append({
                            'prediction': list(pred)[i],
                            'label': list(targets.view_as(pred))[i],
                            'image': img_batch[i]
                            
                        })
            """
            correct += result.sum().item()
            #correct += pred.eq(targets.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 
        100. * correct / len(test_loader.dataset)))
    test_acc.append(100. * correct / len(test_loader.dataset)) 
    return test_loss

#Global functions
def show_summary(model,input_size = (3, 224, 224)):
    summary(model, input_size)
    
def run_model(model, device, optimizer, train_loader, test_loader, train_losses, test_losses, train_acc, test_acc, correct_samples, incorrect_samples, criteria = F.nll_loss, doL1 = 0, doL2 = 0, LAMBDA = 0, EPOCHS = 20,start=0):
    #scheduler = StepLR(optimizer, step_size=8, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1)
    for epoch in range(EPOCHS):
        print("EPOCH:", (start+epoch))
        train_model(model, device, train_loader, optimizer,scheduler, epoch, train_losses, train_acc, criteria, doL1, doL2, LAMBDA)
        last_epoch = False
        if epoch == EPOCHS-1:
          last_epoch = True
        test_model(model, device, test_loader, test_losses, test_acc, criteria,correct_samples,incorrect_samples,last_epoch)
        


