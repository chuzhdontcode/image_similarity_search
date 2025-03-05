from typing import Optional, Union, List
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from transformers import AutoImageProcessor, AutoModel
import numpy as np
from PIL import Image
import fsspec
from torchvision import transforms
from pathlib import Path
import yaml
import os
import requests
from tqdm import tqdm


class BaseEncoder(nn.Module):
    def __init__(self, use_gpu: bool = True):
        super().__init__()
        self.device = torch.device("cuda" if (torch.cuda.is_available() and use_gpu) else "cpu")
        self._load_model_config()

    def _load_model_config(self):
        """Load model configurations from YAML file"""
        config_path = Path(__file__).parent / "config" / "model_urls.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")
        with open(config_path, "r") as f:
            self.model_config = yaml.safe_load(f)

    def encode(self, images: Union[Image.Image, List[Image.Image]]) -> np.ndarray:
        raise NotImplementedError

    def forward(self, x):
        return self.encode(x)


class CLIPEncoder(BaseEncoder):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        super().__init__()
        if model_name not in self.model_config.get("clip", {}):
            raise ValueError(f"Model {model_name} not found in config")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def encode(self, images: Union[Image.Image, List[Image.Image]]) -> np.ndarray:
        if not isinstance(images, list):
            images = [images]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        return image_features.cpu().numpy()


class DINOv2Encoder(BaseEncoder):
    def __init__(self, model_name: str = "facebook/dinov2-base"):
        super().__init__()
        if model_name not in self.model_config.get("dino", {}):
            raise ValueError(f"Model {model_name} not found in config")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    def encode(self, images: Union[Image.Image, List[Image.Image]]) -> np.ndarray:
        if not isinstance(images, list):
            images = [images]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.pooler_output.cpu().numpy()


class SSCDEncoder(BaseEncoder):
    def __init__(self, model_name: str):
        """
        Initialize SSCD encoder with a model name from config
        Args:
            model_name: Name of the model as defined in config/model_urls.yaml
        """
        super().__init__()
        if model_name not in self.model_config.get("sscd", {}):
            raise ValueError(
                f"Model {model_name} not found in config. Available models: {list(self.model_config.get('sscd', {}).keys())}"
            )

        model_info = self.model_config["sscd"][model_name]
        self.model = None
        self.weights_dir = Path(__file__).parent / "weights"
        self.weights_dir.mkdir(exist_ok=True)

        # Download and load the model
        model_path = self.download_weights(model_info["torchvision"], model_name)
        self.load_model(model_path)

        # Define the preprocessing transforms
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        self.transform_288 = transforms.Compose(
            [
                transforms.Resize(288),
                transforms.ToTensor(),
                normalize,
            ]
        )

        self.transform_320 = transforms.Compose(
            [
                transforms.Resize([320, 320]),
                transforms.ToTensor(),
                normalize,
            ]
        )

    def download_weights(self, url: str, model_name: str) -> Path:
        """
        Download weights from URL and cache them locally
        Args:
            url: URL to download weights from
            model_name: Name of the model for local caching
        Returns:
            Path to the downloaded weights file
        """
        # Create filename from model name
        filename = f"{model_name}.pt"
        save_path = self.weights_dir / filename

        # If file already exists, return its path
        if save_path.exists():
            return save_path

        # Download the file with progress bar
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))

        with (
            open(save_path, "wb") as f,
            tqdm(
                desc=f"Downloading {model_name}",
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar,
        ):
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)

        return save_path

    def load_model(self, model_path: Union[str, Path]):
        """Load the SSCD model from a local file path"""
        self.model = torch.jit.load(str(model_path))
        self.model = self.model.to(self.device)
        self.model.eval()

    def encode(self, images: Union[Image.Image, List[Image.Image]], use_320: bool = False) -> np.ndarray:
        if not isinstance(images, list):
            images = [images]

        transform = self.transform_320 if use_320 else self.transform_288

        # Process all images in batch
        batches = []
        for img in images:
            if not isinstance(img, Image.Image):
                raise ValueError("Input must be PIL Image or List[PIL.Image]")
            img = img.convert("RGB")
            batch = transform(img).unsqueeze(0)
            batches.append(batch)

        # Stack all batches
        batch = torch.cat(batches, dim=0).to(self.device)

        # Get embeddings
        with torch.no_grad():
            embeddings = self.model(batch)

        return embeddings.cpu().numpy()
