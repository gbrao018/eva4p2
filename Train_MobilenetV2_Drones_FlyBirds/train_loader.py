
import torch
import torchvision.transforms

#import WingsFolderDataset as wingsDataset
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

root_dir = "/content/Session2Dataset-copy"
# replace with actual mean and std values
normalize = transforms.Normalize(mean=[0.5271, 0.5788, 0.6095], std=[0.1707, 0.1650, 0.1804])

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

