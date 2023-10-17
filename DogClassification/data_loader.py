from typing import List, Tuple
import copy

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_data_loaders(
    images_path: str, 
    length_splits: List = [0.7, 0.1, 0.2], 
    batch_size: int = 128, 
    threads: int = 0, 
    mean: List[int] = [0.485, 0.456, 0.406], 
    std: List[int] = [0.229, 0.224, 0.225],
    seed: int = 23
) -> Tuple[DataLoader, DataLoader, DataLoader, datasets.VisionDataset]:
    train_transform = transforms.Compose([
        #transforms.RandomRotation(degrees=15),
        #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
    ])
    val_transform = train_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
    ])
    dataset = datasets.ImageFolder(root=images_path, transform=train_transform)
    training_set, validation_set, testing_set = random_split(
        dataset=dataset, 
        lengths=length_splits, 
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Change transform of val/test set
    validation_set.dataset = copy.deepcopy(dataset)
    validation_set.dataset.transform = val_transform
    testing_set.dataset = validation_set.dataset
        
    training_set_loader = DataLoader(training_set, batch_size=batch_size, num_workers=threads, shuffle=True)
    validation_set_loader = DataLoader(validation_set, batch_size=batch_size, num_workers=threads, shuffle=True)
    testing_set_loader = DataLoader(testing_set, batch_size=batch_size, num_workers=threads, shuffle=False)
    
    return training_set_loader, validation_set_loader, testing_set_loader, dataset
