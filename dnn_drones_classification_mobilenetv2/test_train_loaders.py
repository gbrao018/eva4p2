import torchvision
import torch
import torchvision.transforms as transforms

def load_dataset():
    train_data_path = '/content/drone-dataset/train'
    test_data_path = '/content/drone-dataset/test'
    train_dataset = torchvision.datasets.ImageFolder(
        root=train_data_path,
        transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=16,
        num_workers=0,
        shuffle=True
    )
    
    test_dataset = torchvision.datasets.ImageFolder(
        root=test_data_path,
        transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=16,
        num_workers=0,
        shuffle=True
    )
    return train_loader, test_loader

train_loader, test_loader = load_dataset()