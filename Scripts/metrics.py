import torch

def calculate_metrics(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    correct = (preds == labels).sum().item()
    accuracy = correct / len(labels)

    # Placeholder for more complex metrics like F1-score
    return {"accuracy": accuracy, "f1_score": 0.0}
