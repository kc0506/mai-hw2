"""Linear Probe model for CLIP vision encoder."""

import torch
import torch.nn as nn
from transformers import CLIPModel


class LinearProbeModel(nn.Module):
    """CLIP vision encoder with frozen backbone and trainable linear classifier."""

    def __init__(self, model_id, num_classes, device="cuda"):
        super().__init__()
        self.device = device
        self.num_classes = num_classes

        # Load CLIP model components
        clip_model = CLIPModel.from_pretrained(model_id)
        self.vision_model = clip_model.vision_model
        self.visual_projection = clip_model.visual_projection

        # Freeze backbone
        for p in self.vision_model.parameters():
            p.requires_grad = False
        for p in self.visual_projection.parameters():
            p.requires_grad = False

        # Create classifier head
        embedding_dim = self.visual_projection.out_features
        self.head = nn.Linear(embedding_dim, num_classes)

        # Move to device
        self.vision_model.to(device)
        self.visual_projection.to(device)
        self.head.to(device)

    def forward(self, pixel_values):
        """Forward pass through vision encoder and classifier.

        Args:
            pixel_values: Batch of images [B, C, H, W]

        Returns:
            torch.Tensor: Class logits [B, num_classes]
        """
        # Extract features from frozen vision encoder
        with torch.no_grad():
            vision_outputs = self.vision_model(pixel_values=pixel_values)
            image_embeds = vision_outputs.last_hidden_state[:, 0, :]  # CLS token
            image_embeds = self.visual_projection(image_embeds)

        # Classification
        logits = self.head(image_embeds)
        return logits

    def get_trainable_parameters(self):
        """Get list of trainable parameters."""
        return self.head.parameters()

    def save_checkpoint(self, path):
        """Save model checkpoint."""
        torch.save({
            "head_state_dict": self.head.state_dict(),
            "num_classes": self.num_classes,
        }, path)

    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.head.load_state_dict(checkpoint["head_state_dict"])
