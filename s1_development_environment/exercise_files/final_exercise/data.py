import torch
import pathlib
import os

datadir = "../../../data/corruptmnist"

print(os.getcwd())
def mnist():
    """Return train and test dataloaders for MNIST."""
    # exchange with the corrupted mnist dataset
    #train = torch.randn(50000, 784)
    #test = torch.randn(10000, 784)

    train_data = [f"{datadir}/train_images_{i}.pt" for i in range(0,6)]
    train_labels = [f"{datadir}/train_target_{i}.pt" for i in range(0,6)]
    trainset = CustomMnistDataset(train_data, train_labels)
    testset = CustomMnistDataset([f"{datadir}/test_images.pt"], [f"{datadir}/test_target.pt"])

    train = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    test = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
    return train, test

class CustomMnistDataset():
    def __init__(self, trainfiles, labelfiles, transform=None, target_transform=None):
        self.datadir = datadir
        self.labels = self.load(labelfiles)
        self.data = self.load(trainfiles)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        datarow = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            datarow = self.transform(datarow)
        if self.target_transform:
            label = self.target_transform(label)
        return datarow, label
    
    def load(self, file_list):
        result = torch.load(file_list[0])
        for file in file_list[1:]:
            result = torch.cat((result, torch.load(file)))
        return result