import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor


class MultiTaskHead(nn.Module):
    """Head for multiple task outputs."""

    def __init__(self, input_dim, num_classes_task1, num_classes_task2):
        super(MultiTaskHead, self).__init__()
        self.task1_head = nn.Linear(input_dim, num_classes_task1)
        self.task2_head = nn.Linear(input_dim, num_classes_task2)

    def forward(self, x):
        task1_output = self.task1_head(x)
        task2_output = self.task2_head(x)
        return task1_output, task2_output


class AttentionFusion(nn.Module):
    """Attention layer for multi-modal data fusion."""

    def __init__(self, input_dim):
        super(AttentionFusion, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(input_dim, 1))

    def forward(self, eeg_features, clip_features):
        combined = torch.cat((eeg_features, clip_features), dim=1)
        attention_scores = F.softmax(torch.matmul(combined, self.attention_weights), dim=1)
        fused_output = torch.mul(combined, attention_scores)
        return fused_output


class EngHealthCLIP(nn.Module):
    def __init__(self, eeg_input_dim, hidden_dim, output_dim_task1, output_dim_task2):
        super(EngHealthCLIP, self).__init__()

        # Initialize CLIP model
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

        # EEG Encoder
        self.eeg_encoder = nn.Sequential(
            nn.Linear(eeg_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Attention-based Fusion Layer
        self.attention_fusion = AttentionFusion(hidden_dim + self.clip_model.config.hidden_size)

        # Multi-task heads
        self.multi_task_head = MultiTaskHead(hidden_dim + self.clip_model.config.hidden_size,
                                             output_dim_task1, output_dim_task2)

        # Loss functions for multi-task learning
        self.task1_loss = nn.CrossEntropyLoss()
        self.task2_loss = nn.MSELoss()

    def forward(self, eeg_data, text_data, image_data):
        # Encode EEG data
        eeg_features = self.eeg_encoder(eeg_data)

        # Encode text and image data with CLIP
        clip_text_features = self.clip_model.get_text_features(text_data)
        clip_image_features = self.clip_model.get_image_features(image_data)
        clip_features = torch.cat((clip_text_features, clip_image_features), dim=1)

        # Attention-based feature fusion
        fused_features = self.attention_fusion(eeg_features, clip_features)

        # Multi-task outputs
        task1_output, task2_output = self.multi_task_head(fused_features)
        return task1_output, task2_output

    def compute_loss(self, task1_pred, task2_pred, task1_target, task2_target, task1_weight=0.5):
        """Compute multi-task loss with weighted sum."""
        loss1 = self.task1_loss(task1_pred, task1_target)
        loss2 = self.task2_loss(task2_pred, task2_target)
        total_loss = task1_weight * loss1 + (1 - task1_weight) * loss2
        return total_loss


if __name__ == "__main__":
    # Example inputs
    batch_size = 16
    eeg_input_dim = 128
    text_length = 77  # CLIP default
    image_dim = (3, 224, 224)  # CLIP default

    hidden_dim = 256
    num_classes_task1 = 5  # Example: stress levels
    num_classes_task2 = 1  # Example: fatigue score (regression)

    model = EngHealthCLIP(eeg_input_dim, hidden_dim, num_classes_task1, num_classes_task2)

    # Random synthetic data for testing
    eeg_data = torch.randn(batch_size, eeg_input_dim)
    text_data = torch.randint(0, 49408, (batch_size, text_length))  # Random text token indices
    image_data = torch.randn(batch_size, *image_dim)  # Random image tensor

    task1_target = torch.randint(0, num_classes_task1, (batch_size,))
    task2_target = torch.rand(batch_size)

    # Forward pass
    task1_output, task2_output = model(eeg_data, text_data, image_data)

    # Compute loss
    loss = model.compute_loss(task1_output, task2_output, task1_target, task2_target)
    print(f"Task 1 Output Shape: {task1_output.shape}")
    print(f"Task 2 Output Shape: {task2_output.shape}")
    print(f"Loss: {loss.item()}")
