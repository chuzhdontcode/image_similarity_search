from typing import List, Union, Optional, Tuple
import numpy as np
import torch
import faiss
from PIL import Image
from tqdm.auto import tqdm
from .models import CLIPEncoder, SSCDEncoder, DINOv2Encoder


class ImageSearchEngine:
    def __init__(
        self,
        model_type: str = "clip",
        use_gpu: bool = False,
        dimension: Optional[int] = None,
        index_type: str = "l2",
        metric: str = "l2",
        normalize_features: bool = False,
        model_name: Optional[str] = None,
    ):
        """
        Initialize the image search engine.

        Args:
            model_type: One of 'clip', 'sscd', or 'dinov2'
            use_gpu: Whether to use GPU for model encoding (FAISS remains on CPU)
            dimension: Feature dimension (optional, determined automatically)
            index_type: Type of FAISS index ('l2' or 'ivf')
            metric: Distance metric to use ('l2' or 'inner_product')
            normalize_features: Whether to L2 normalize feature vectors
            model_name: Specific model name to use (required for SSCD, optional for others)
        """
        self.model_type = model_type.lower()
        self.use_gpu = use_gpu and torch.cuda.is_available()  # Only affects model
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric.lower()
        self.normalize_features = normalize_features

        if self.metric not in ["l2", "inner_product"]:
            raise ValueError(f"Unknown metric: {metric}. Must be 'l2' or 'inner_product'")

        # Initialize the encoder
        if self.model_type == "clip":
            self.encoder = CLIPEncoder(model_name if model_name else "openai/clip-vit-base-patch32")
        elif self.model_type == "sscd":
            if model_name is None:
                raise ValueError("model_name parameter is required for SSCD encoder")
            self.encoder = SSCDEncoder(model_name=model_name)
        elif self.model_type == "dinov2":
            self.encoder = DINOv2Encoder(model_name if model_name else "facebook/dinov2-base")
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.index = None
        self.image_ids = []

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """L2 normalize feature vectors."""
        if self.normalize_features:
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            features = features / norms
        return features

    def _create_index(self, dimension: int):
        """Create FAISS index based on the specified type and metric."""
        if self.index_type == "l2":
            if self.metric == "l2":
                index = faiss.IndexFlatL2(dimension)
            else:  # inner_product
                index = faiss.IndexFlatIP(dimension)
        elif self.index_type == "ivf":
            if self.metric == "l2":
                quantizer = faiss.IndexFlatL2(dimension)
            else:  # inner_product
                quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, 100)  # 100 centroids
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        return index

    def add_images(self, images: List[Image.Image], image_ids: Optional[List[str]] = None):
        """
        Add images to the search index.

        Args:
            images: List of PIL Images
            image_ids: Optional list of image identifiers
        """
        # Process images in smaller batches to show progress
        batch_size = 32
        all_features = []
        
        for i in tqdm(range(0, len(images), batch_size), desc="Encoding images"):
            batch = images[i:i + batch_size]
            features = self.encoder.encode(batch)
            all_features.append(features)
        
        # Combine all features
        features = np.vstack(all_features)
        features = self._normalize_features(features)

        if self.index is None:
            self.dimension = features.shape[1]
            self.index = self._create_index(self.dimension)
            if self.index_type == "ivf":
                print("Training IVF index...")
                self.index.train(features)

        print("Adding features to index...")
        self.index.add(features)

        if image_ids is None:
            start_idx = len(self.image_ids)
            image_ids = [str(i) for i in range(start_idx, start_idx + len(images))]

        self.image_ids.extend(image_ids)

    def search(self, query_image: Union[Image.Image, List[Image.Image]], k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for similar images.

        Args:
            query_image: Query image(s)
            k: Number of results to return

        Returns:
            List of (image_id, distance) tuples
        """
        if self.index is None:
            raise RuntimeError("No images added to the index yet")

        # Handle batch processing with progress bar
        if isinstance(query_image, list):
            batch_size = 32
            all_query_features = []
            
            for i in tqdm(range(0, len(query_image), batch_size), desc="Encoding query images"):
                batch = query_image[i:i + batch_size]
                features = self.encoder.encode(batch)
                all_query_features.append(features)
            
            query_features = np.vstack(all_query_features)
        else:
            query_features = self.encoder.encode(query_image)

        query_features = self._normalize_features(query_features)
        distances, indices = self.index.search(query_features, k)

        results = []
        for idx_list, dist_list in zip(indices, distances):
            batch_results = []
            for idx, dist in zip(idx_list, dist_list):
                if idx < len(self.image_ids):  # Ensure valid index
                    # For inner product, higher values are better, so we negate the distance
                    if self.metric == "inner_product":
                        dist = -dist
                    batch_results.append((self.image_ids[idx], float(dist)))
            results.append(batch_results)

        return results[0] if not isinstance(query_image, list) else results

    def save_index(self, filepath: str):
        """Save the FAISS index to disk."""
        if self.index is not None:
            faiss.write_index(self.index, filepath)

    def load_index(self, filepath: str):
        """Load the FAISS index from disk."""
        self.index = faiss.read_index(filepath)
