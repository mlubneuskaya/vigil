"""
DINO v3 Embedding Utility

Extracts DINO v3 embeddings for image regions (object masks).
"""

import torch
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoImageProcessor


class DINOv3Embedder:
    def __init__(
        self,
        model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        device: str = "cuda",
    ):
        self.device = device
        print(f"Loading {model_name}...")
        self.model = AutoModel.from_pretrained(model_name, device_map="auto")
        self.model.eval()
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        print(f"DINO v3 loaded successfully on {device}")

    def get_embedding(self, images: list[Image.Image]) -> np.ndarray:
        """
        Computes embeddings for a list of object crops in parallel.
        Args:
            crops: List of PIL Images (each image is a cropped object)
        Returns:
            embeddings: numpy array of shape (batch_size, embedding_dim)
        """
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            cls_tokens = outputs.last_hidden_state[:, 0, :]
            cls_tokens = torch.nn.functional.normalize(cls_tokens, p=2, dim=1)

        return cls_tokens
