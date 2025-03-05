from collections import defaultdict
from pathlib import Path
import shutil
from typing import Dict, Set, Optional
from PIL import Image
from tqdm.auto import tqdm
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster

from .search_engine import ImageSearchEngine


class ImageClusterer:
    def __init__(
        self,
        model_type: str = "clip",
        use_gpu: bool = False,
        threshold: float = 0.22,
        batch_size: int = 192,
        model_name: Optional[str] = None,
        normalize_features: bool = True,
    ):
        """Initialize the image clusterer.

        Args:
            model_type: One of 'clip', 'sscd', or 'dinov2'
            use_gpu: Whether to use GPU for model encoding
            threshold: Distance threshold for hierarchical clustering
            batch_size: Batch size for processing images
            model_name: Specific model name to use (required for SSCD)
            normalize_features: Whether to normalize feature vectors
        """
        self.search_engine = ImageSearchEngine(
            model_type=model_type,
            use_gpu=use_gpu,
            normalize_features=normalize_features,
            model_name=model_name,
        )
        self.threshold = threshold
        self.batch_size = batch_size

    def process_images(self, image_directory: str) -> Dict[int, Set[str]]:
        """Process images in a directory and return clusters.

        Args:
            image_directory: Path to directory containing images (including subdirectories)

        Returns:
            Dictionary mapping cluster IDs to sets of image paths
        """
        image_directory = Path(image_directory)
        allowed_extensions = {".jpeg", ".jpg", ".png", ".webp"}

        # Get all valid image paths recursively
        image_paths = [p for p in image_directory.rglob("*") if p.suffix.lower() in allowed_extensions]

        if not image_paths:
            raise ValueError(f"No valid images found in {image_directory} or its subdirectories")

        # Process images in batches
        batch_images = []
        valid_paths = []
        damaged_paths = set()

        print("Loading and validating images...")
        for i in tqdm(range(0, len(image_paths), self.batch_size), desc="Processing image batches"):
            batch_paths = image_paths[i : i + self.batch_size]
            current_batch = []
            current_valid_paths = []

            # Load images
            for path in batch_paths:
                try:
                    img = Image.open(path)
                    img.load()  # Verify the image can be loaded
                    current_batch.append(img)
                    current_valid_paths.append(path)
                except Exception as e:
                    print(f"\nError processing image {path}: {e}")
                    damaged_paths.add(path)
                    continue

            if current_batch:
                batch_images.extend(current_batch)
                valid_paths.extend(current_valid_paths)

        if not valid_paths:
            raise ValueError("No valid images could be processed")

        # Add images to the search engine's FAISS index
        self.search_engine.add_images(batch_images, [str(p) for p in valid_paths])
        
        # Use the search engine to compute all pairwise distances
        print("Computing distance matrix using FAISS...")
        num_images = len(batch_images)
        
        # Get all pairwise distances using the search engine's FAISS index
        distances = []
        for i in tqdm(range(num_images), desc="Computing pairwise distances"):
            # Search for all other images using the current image as query
            results = self.search_engine.search(batch_images[i], k=num_images)
            # Extract only the distances for the upper triangular part
            for j, (_, dist) in enumerate(results):
                if j > i:  # We only need upper triangular part
                    distances.append(dist)
                    print(dist)

        # Perform hierarchical clustering
        print("Performing hierarchical clustering...")
        Z = linkage(distances, method="average", optimal_ordering=True)
        labels = fcluster(Z, t=self.threshold, criterion="distance")

        # Group images by cluster
        clusters: Dict[int, Set[str]] = defaultdict(set)
        for path, label in zip(valid_paths, labels):
            clusters[int(label)].add(str(path))

        return clusters

    def organize_clusters(
        self, clusters: Dict[int, Set[str]], output_directory: str, min_cluster_size: int = 2
    ) -> None:
        """Organize clustered images into output directories.

        Args:
            clusters: Dictionary mapping cluster IDs to sets of image paths
            output_directory: Base directory to output organized clusters
            min_cluster_size: Minimum number of images to form a cluster
        """
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)

        # Organize images into clusters
        all_clustered_images = set()
        print("Organizing clusters...")
        for idx in tqdm(clusters.keys(), desc="Processing clusters"):
            image_paths = clusters[idx]
            if len(image_paths) >= min_cluster_size:
                cluster_dir = output_directory / f"cluster_{idx}"
                cluster_dir.mkdir(parents=True, exist_ok=True)

                for image_path in image_paths:
                    source = Path(image_path)
                    destination = cluster_dir / source.name
                    shutil.copy(source, destination)
                    all_clustered_images.add(str(source))

        # Move unique images (not in any cluster or in small clusters)
        unique_dir = output_directory / "unique"
        unique_dir.mkdir(parents=True, exist_ok=True)

        all_images = {
            str(p)
            for p in Path(output_directory).parent.glob("*")
            if p.suffix.lower() in {".jpeg", ".jpg", ".png", ".webp"}
        }

        unique_images = all_images - all_clustered_images
        print("Processing unique images...")
        for image_path in tqdm(unique_images, desc="Moving unique images"):
            source = Path(image_path)
            destination = unique_dir / source.name
            shutil.copy(source, destination)
