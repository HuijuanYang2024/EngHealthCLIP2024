import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.model import EngHealthCLIP
from data.data_loader import MultimodalDataset
from utils.logging import setup_logger
from utils.metrics import calculate_metrics

logger = setup_logger("train.log")


def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize dataset and dataloader
    dataset = MultimodalDataset(config["data"]["train_path"])
    dataloader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=True)

    # Initialize model, loss, and optimizer
    model = EngHealthCLIP(config["model"]["eeg_dim"], config["model"]["hidden_dim"], config["model"]["output_dim"]).to(
        device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config["training"]["epochs"]):
        model.train()
        epoch_loss = 0
        for batch in dataloader:
            eeg_data, text_data, image_data, labels = batch
            eeg_data, text_data, image_data, labels = eeg_data.to(device), text_data.to(device), image_data.to(
                device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(eeg_data, text_data, image_data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        logger.info(f"Epoch {epoch + 1}/{config['training']['epochs']}, Loss: {epoch_loss:.4f}")
