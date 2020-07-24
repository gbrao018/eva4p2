
import torch
import torchvision.transforms

#import WingsFolderDataset as wingsDataset
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

root_dir = "/content/Session2Dataset-copy"
# replace with actual mean and std values
normalize = transforms.Normalize(mean=[149.17579724, 143.51813416, 136.34473418],
                                        std=[10.918, 10.54722837, 9.7497292])

tr = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(),
    transforms.ToTensor()
    normalize,
    transforms.RandomErasing()
])

train_set = WingsFolderDataset("/content/Session2Dataset-train", transform = tr)



train_loader = torch.utils.data.DataLoader(train_set, batch_size=64,
                                          shuffle=True, num_workers=1,pin_memory=False)

test_transform = transforms.Compose([transforms.ToTensor(),normalize])
test_set = WingsFolderDataset("/content/Session2Dataset-test", transform = tr)('/content/Dataset_224/', size = 100000, test=True, transform = test_transform)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, num_workers=1)
