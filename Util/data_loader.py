import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset

def get_transform(img_size=(224, 224)):
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size[0]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

class UnlabeledDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0  # Return a dummy label

def load_initial_subset(labelled_data_dir, batch_size=32, initial_sample_size=100):
    transform = get_transform()
    all_dataset = datasets.ImageFolder(root=labelled_data_dir, transform=transform)
    indices = np.random.choice(len(all_dataset), initial_sample_size, replace=False)
    initial_dataset = Subset(all_dataset, indices)
    return DataLoader(initial_dataset, batch_size=batch_size, shuffle=True), all_dataset

def create_unlabeled_loader(unlabeled_data_dir, batch_size=32, img_size=(224, 224)):
    image_paths = [os.path.join(root, name)
                   for root, dirs, files in os.walk(unlabeled_data_dir)
                   for name in files
                   if name.endswith((".png", ".jpg", ".jpeg"))]
    dataset = UnlabeledDataset(image_paths=image_paths, transform=get_transform(img_size))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
