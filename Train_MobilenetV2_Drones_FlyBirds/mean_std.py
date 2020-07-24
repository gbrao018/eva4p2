import torch

def mean_and_std(dataset):
	''' compute the mean and std of the given dataset'''
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
	mean = torch.zeros(3)
	std = torch.zeros(3)
	print('Computing mean and std from dataloader')
	for inputs, target in dataloader:
		for i in range(3):
			mean[i] + = inputs[:,i,:, :].mean()
			std[i] + = inputs[:,i,:, :].std()
	mean.div_(len(dataset))
	std.div_(len(dataset))
	return mean, std