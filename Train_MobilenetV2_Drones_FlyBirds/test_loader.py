import torch
import torchvision.transforms

normalize = transforms.Normalize(mean=[0.5271, 0.5788, 0.6095], std=[0.1707, 0.1650, 0.1804])

test_transform = transforms.Compose([transforms.ToTensor(),normalize])

test_set = WingsFolderDataset("/content/Session2Dataset-test", transform = test_transform)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, num_workers=1)
