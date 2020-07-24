test_transform = transforms.Compose([transforms.ToTensor(),normalize])
test_set = WingsFolderDataset("/content/Session2Dataset-test", transform = tr)('/content/Dataset_224/', size = 100000, test=True, transform = test_transform)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, num_workers=1)
