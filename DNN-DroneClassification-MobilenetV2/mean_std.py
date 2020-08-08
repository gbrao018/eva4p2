import torch
from torch.utils.data import DataLoader, Dataset

#mean= tensor([0.5271, 0.5788, 0.6095])
#std= tensor([0.1707, 0.1650, 0.1804])
#(tensor([0.5271, 0.5788, 0.6095]), tensor([0.1707, 0.1650, 0.1804]))

import torch
from torchvision import transforms

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#Assuming test data represent the full set of train data. So we use normalized values of traion data in test data also.

train_root_dir = "/content/drone-dataset/train"
processed_train_root_dir = "/content/processed-drone-dataset/train"

tr = transforms.Compose([
    #transforms.RandomHorizontalFlip(),
    #transforms.RandomCrop(),
    transforms.ToTensor()
    #normalize,
    #transforms.RandomErasing()
])


train_data_set = preprocess_images(train_root_dir, processed_train_root_dir, transform = tr)

def mean_std(dataset):
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                          shuffle=False, num_workers=1)

  mean = torch.zeros(3)
  std = torch.zeros(3)
  print('Computing mean and std from dataloader')
  for inputs in dataloader:
	  for i in range(3):
		  mean[i] = mean[i] + inputs[:,i,:, :].mean()
		  std[i]  = std[i] + inputs[:,i,:, :].std()
  mean.div_(len(dataset))
  std.div_(len(dataset))
  print('mean=',mean)
  print('std=',std)
  return mean, std

print('calculating mean,std of train dataset')
mean_std(train_data_set)

test_root_dir = "/content/drone-dataset/test"
processed_test_root_dir = "/content/processed-drone-dataset/test"

test_data_set = PreprocessImages(test_root_dir, processed_test_root_dir, transform = tr)

print('calculating mean,std of test dataset')
mean_std(test_data_set)
