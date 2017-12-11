import numpy as np

from torch.utils import data 
import torch
from torchvision import datasets, transforms

class DynamicSampler(object):
    def __init__(self, max_size=100):
        self.next_batch = [0]
        self.max_size = max_size

    def select_sample(self, indList):
        self.next_batch = indList

    def __iter__(self):
        return iter(self.next_batch)

    def __len__(self):
        return self.max_size

def get_batch(dataset, indices=None, num_workers=2):
    sampler = DynamicSampler(len(indices))
    loader = data.DataLoader(dataset, 
                  batch_size=len(indices), 
                  sampler=sampler)
    
    sampler.select_sample(indices)

    return iter(loader).next()

if __name__ == '__main__':
    train_dataset = datasets.MNIST('./mnist_data', train=True, download=True,
               transform=transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,))]))
    indices = np.arange(_i, _i+_BATCH_SIZE)
    batch = get_batch(train_dataset, indices)
    print(batch[0].size())
    print(batch[1].size())