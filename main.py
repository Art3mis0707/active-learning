import torch
import torch.optim as optim
from models.pretrain_model import PretrainedEfficientNet
from models.active_learner import uncertainty_sampling
from utils.data_loader import load_labelled_data, load_unlabelled_data

def main():
   
    num_classes = 10
    labelled_data_dir = 'path/to/labelled/data'
    unlabelled_data_dir = 'path/to/unlabelled/data'
    num_epochs = 5  # Number of epochs for initial training
    num_samples_per_round = 10  # Number of samples to query in each active learning round

    # Initialize the model
    model = PretrainedEfficientNet(num_classes=num_classes)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

   
    labelled_loader = load_labelled_data(labelled_data_dir)

    # Initial training phase
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in labelled_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs} completed')

    # Active learning rounds
    for round in range(5):  # Number of rounds
        print(f'Active learning round {round+1}')
        
        # Select samples using active learning
        informative_samples = uncertainty_sampling(model, unlabelled_data_dir, num_samples=num_samples_per_round)

        # Implement human-in-the-loop annotation tool display the images and ask the user to provide labels
        for sample in informative_samples:
            image = sample[0]
            label = input(f"Please provide a label for the image: {image}")
            sample[1] = label

        
        # Add the informative samples to the labelled dataset
        labelled_loader.dataset.samples.extend(informative_samples)

        # Re-train the model with the augmented dataset
        # Training loop (same as initial training)
        model.train()
        for epoch in range(num_epochs):
            for inputs, labels in labelled_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch+1}/{num_epochs} in round {round+1} completed')

    print('Active learning process completed')

if __name__ == "__main__":
    main()
