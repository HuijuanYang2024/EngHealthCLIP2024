import torch
from model.model import EngHealthCLIP
from data.data_loader import MultimodalDataset
from torch.utils.data import DataLoader
from utils.metrics import calculate_metrics

def evaluate_model(config, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MultimodalDataset(config["data"]["test_path"])
    dataloader = DataLoader(dataset, batch_size=config["evaluation"]["batch_size"], shuffle=False)

    model = EngHealthCLIP(config["model"]["eeg_dim"], config["model"]["hidden_dim"], config["model"]["output_dim"]).to(device)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    total_metrics = {"accuracy": 0, "f1_score": 0}
    with torch.no_grad():
        for batch in dataloader:
            eeg_data, text_data, image_data, labels = batch
            eeg_data, text_data, image_data, labels = eeg_data.to(device), text_data.to(device), image_data.to(device), labels.to(device)

            outputs = model(eeg_data, text_data, image_data)
            metrics = calculate_metrics(outputs, labels)
            for key in metrics:
                total_metrics[key] += metrics[key]

    print({key: val / len(dataloader) for key, val in total_metrics.items()})
