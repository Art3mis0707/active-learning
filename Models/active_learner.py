import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

def uncertainty_sampling(model, data_dir, num_samples=10):
    # Load the unlabelled data
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model.eval()
    uncertainties = []

    with torch.no_grad():
        for inputs, _ in dataloader:
            outputs = model(inputs)
            # Calculate uncertainty (e.g., using entropy)
            entropy = -torch.sum(outputs.softmax(1) * torch.log(outputs.softmax(1) + 1e-5), dim=1)
            uncertainties.append((inputs, entropy.item()))

    # Select the samples with the highest uncertainty
    selected_samples = sorted(uncertainties, key=lambda x: x[1], reverse=True)[:num_samples]
    return [sample[0] for sample in selected_samples]


# model = PretrainedEfficientNet(num_classes=10)
# informative_samples = uncertainty_sampling(model, 'path/to/unlabelled/data')
