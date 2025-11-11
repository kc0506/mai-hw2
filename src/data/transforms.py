"""Image transforms for CLIP models."""


class CLIPTransform:
    """Transform that processes images for CLIP."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, image):
        """Process a single PIL image for CLIP.

        Args:
            image: PIL Image

        Returns:
            torch.Tensor: Processed pixel values
        """
        return self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
