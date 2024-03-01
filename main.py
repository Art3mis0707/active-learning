import torch
import torch.nn as nn
import torch.optim as optim
import models.active_learner as active 
import util.data_loader as loader 
from torch.utils.data import Subset
from torchvision.models import resnet18, ResNet18_Weights




def train_model(model, criterion, optimizer, dataloader, device,  epochs=5):
    model.train()
    print("Initial training begins")
    for epoch in range(epochs):
        for images, labels in dataloader:
            # Adjust for CUDA availability
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    print("Initial training is complete")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 256
    labelled_data_dir = 'C:\\Users\\HP\\OneDrive\\Desktop\\ALTL\\caltech256_extracted'
    unlabeled_data_dir = 'C:\\Users\\HP\\OneDrive\\Desktop\\ALTL\\caltech256_extracted'

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    initial_loader, all_labeled_dataset = loader.load_initial_subset(labelled_data_dir, batch_size=32, initial_sample_size=100)
    train_model(model, criterion, optimizer, initial_loader, device, epochs=5)

    print("Beginning Active Learning selection")
    for cycle in range(5):
        print('Epoch :', cycle)

        print('Loading unlabelled data')
        unlabeled_loader = loader.create_unlabeled_loader(unlabeled_data_dir, batch_size=32)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Selecting samples for labelling using active learning")
        selected_indices = active.select_samples_for_labeling(model, unlabeled_loader, device, num_samples=10)

        
        print('Simulate labeling by adding selected samples to the labeled dataset')
        selected_subset = Subset(unlabeled_loader.dataset, selected_indices)
        all_labeled_dataset = loader.ConcatDataset([all_labeled_dataset, selected_subset])
        updated_loader = loader.DataLoader(all_labeled_dataset, batch_size=32, shuffle=True)
        
        print("Retraining the model after the updation")
        train_model(model, criterion, optimizer, updated_loader, device, epochs=5)
    print('Active Learning selection complete')

if __name__ == "__main__":
    main()
