from typing import Optional, Union, List
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from transformers import AutoImageProcessor, AutoModel
import numpy as np
from PIL import Image

class BaseEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def encode(self, images: Union[Image.Image, List[Image.Image]]) -> np.ndarray:
        raise NotImplementedError
        
class CLIPEncoder(BaseEncoder):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
    def encode(self, images: Union[Image.Image, List[Image.Image]]) -> np.ndarray:
        if not isinstance(images, list):
            images = [images]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        return image_features.cpu().numpy()

class SSCDEncoder(BaseEncoder):
    def __init__(self, model_name: str = "microsoft/swin-base-patch4-window7-224"):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
    def encode(self, images: Union[Image.Image, List[Image.Image]]) -> np.ndarray:
        if not isinstance(images, list):
            images = [images]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.pooler_output.cpu().numpy()

class DINOv2Encoder(BaseEncoder):
    def __init__(self, model_name: str = "facebook/dinov2-base"):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
    def encode(self, images: Union[Image.Image, List[Image.Image]]) -> np.ndarray:
        if not isinstance(images, list):
            images = [images]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.pooler_output.cpu().numpy()