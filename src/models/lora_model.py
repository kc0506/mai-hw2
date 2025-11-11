"""LoRA fine-tuned CLIP vision encoder."""

import torch
import torch.nn as nn
from transformers import CLIPModel
from peft import LoraConfig, get_peft_model


class LoRAModel(nn.Module):
    """CLIP vision encoder with LoRA adapters and trainable classifier."""

    def __init__(
        self,
        model_id,
        num_classes,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        device="cuda"
    ):
        super().__init__()
        self.device = device
        self.num_classes = num_classes

        # Load CLIP model components
        clip_model = CLIPModel.from_pretrained(model_id)
        vision_model = clip_model.vision_model
        self.visual_projection = clip_model.visual_projection

        # Configure LoRA
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            # target_modules=["q_proj", "v_proj"],
            target_modules=["q_proj"],
            lora_dropout=lora_dropout,
            bias="none"
            # DO NOT use task_type for vision models
        )

        # Wrap vision model with PEFT
        self.vision_model_lora = get_peft_model(vision_model, lora_config)

        # Freeze projection layer
        for p in self.visual_projection.parameters():
            p.requires_grad = False

        # Create classifier head
        embedding_dim = self.visual_projection.out_features
        self.head = nn.Linear(embedding_dim, num_classes)

        # Move to device
        self.vision_model_lora.to(device)
        self.visual_projection.to(device)
        self.head.to(device)

    def forward(self, pixel_values):
        """Forward pass through LoRA vision encoder and classifier.

        Args:
            pixel_values: Batch of images [B, C, H, W]

        Returns:
            torch.Tensor: Class logits [B, num_classes]
        """

        # Extract features from LoRA-enhanced vision encoder
        vision_outputs = self.vision_model_lora(pixel_values=pixel_values)
        image_embeds = vision_outputs.last_hidden_state[:, 0, :]  # CLS token

        # Project embeddings (frozen)
        with torch.no_grad():
            image_embeds = self.visual_projection(image_embeds)

        # Classification
        logits = self.head(image_embeds)
        return logits

    def get_trainable_parameters(self):
        """Get list of trainable parameters."""
        return [x for x in self.vision_model_lora.parameters() if x.requires_grad] + list(self.head.parameters())

    def print_trainable_parameters(self):
        """Print trainable parameter statistics."""
        self.vision_model_lora.print_trainable_parameters()

    def save_checkpoint(self, path):
        """Save model checkpoint."""
        torch.save({
            "lora_state_dict": self.vision_model_lora.state_dict(),
            "head_state_dict": self.head.state_dict(),
            "num_classes": self.num_classes,
        }, path)

    def save_lora_adapters(self, path):
        """Save only LoRA adapters (lightweight)."""
        self.vision_model_lora.save_pretrained(path)

    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.vision_model_lora.load_state_dict(checkpoint["lora_state_dict"])
        self.head.load_state_dict(checkpoint["head_state_dict"])
