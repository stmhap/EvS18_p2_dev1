from torchvision import datasets, transforms
from utils import one_hot_encode
import pytorch_lightning as pl
import torch
import random

class CIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(CIFAR10, self).__init__(root, train, transform, target_transform, download)

    def __getitem__(self, index):
        img, target = super(CIFAR10, self).__getitem__(index)              
        # For VAE encoder, we train model for images with correct label in OHE
        target_ohe = one_hot_encode(target,10)
        # random_OHE to train model for images with incorrect label
        random_nr =  random.randint(1, 100) % 10
        target_ohe_random = one_hot_encode(random_nr,10)
        return img, target_ohe, target_ohe_random
    
class CIFAR10_DataModule(pl.LightningDataModule):
    def __init__(self, data_dir='data', batch_size=64):
        super().__init__()       
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self):
        # Resize to 32 as Resnet18 accepts input image of size 32
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        # Get custom CIFAR10 with OHE of target
        self.cifar10_train = CIFAR10(root=self.data_dir, train=True, transform=transform, download=True)
        self.cifar10_val = CIFAR10(root=self.data_dir, train=False, transform=transform)

    def setup(self,stage):
        pass
                 

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.cifar10_train, self.batch_size, shuffle=True)
    
    def test_dataloader(self):
         return torch.utils.data.DataLoader(self.cifar10_val, self.batch_size, shuffle=False)
    
    def val_dataloader(self):
         return torch.utils.data.DataLoader(self.cifar10_val, self.batch_size, shuffle=False)