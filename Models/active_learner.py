import torch
from torchvision import transforms
import numpy as np

def select_samples_for_labeling(model, dataloader, device, num_samples=10):
    model.eval()
    uncertainties = []
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            top_probs, _ = torch.topk(probs, 1)
            uncertainty = 1.0 - top_probs.squeeze()
            uncertainties.extend(uncertainty.cpu().numpy())

            # Debugging prints
            print("Predictions:", preds)
            print("Top probabilities:", top_probs)
            print("Uncertainty:", uncertainty)

    uncertainties = np.array(uncertainties)
    selected_indices = np.argsort(uncertainties)[-num_samples:]
    
    # Debugging print
    print("Selected indices:", selected_indices)
    print("Uncertainties of selected:", uncertainties[selected_indices])
    
    return selected_indices


