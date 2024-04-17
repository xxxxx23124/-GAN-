import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch

def get_dataset(dataroot, size=64):
    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
                                   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                               ]),

                               )
    return dataset
def get_MNIST(dataroot):
    dataset = dset.MNIST(root=dataroot,
                         train=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             #transforms.RandomHorizontalFlip(),
                             transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.BICUBIC),
                             transforms.Normalize((0.5), (0.5))
                         ]),
                         download=True,
                         )
    return dataset

def get_dataloader(dataset, batch_size):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True)
    return dataloader