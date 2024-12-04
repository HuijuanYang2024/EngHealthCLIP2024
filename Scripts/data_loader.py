import torch
from torch.utils.data import Dataset
from transformers import CLIPProcessor

class MultimodalDataset(Dataset):
    def __init__(self, data_path):
        self.data = torch.load(data_path)  # Load preprocessed data
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        eeg_data, text, image_path, label = self.data[idx]
        image = self.processor(images=image_path, return_tensors="pt")["pixel_values"]
        text = self.processor(text=text, return_tensors="pt")["input_ids"]
        return eeg_data, text, image, label
