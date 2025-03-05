# Image Similarity Search Engine

A flexible image processing library that provides two main functionalities:
1. Image Similarity Search: Find similar images using state-of-the-art embedding models
2. Image Clustering: Group similar images using hierarchical clustering

Both features support multiple embedding models (CLIP, SSCD, DINOv2) and both CPU and GPU operations using FAISS.

## Installation

For CPU-only usage:
```bash
pip install .
```

For GPU support:
```bash
pip install ".[gpu]"
```

## Supported Models
Refer to [Model List](image_similarity_search/config/model_urls.yaml)

## Image Similarity Search

The `ImageSearchEngine` class provides fast and efficient similarity search capabilities:

```python
from PIL import Image
from image_similarity_search import ImageSearchEngine

# Initialize the search engine
search_engine = ImageSearchEngine(
    model_type="clip",  # or "sscd" or "dinov2"
    use_gpu=True,  # Set to False for CPU-only
    index_type="l2"  # or "ivf" for larger datasets
)

# Add images to the index
images = [Image.open("image1.jpg"), Image.open("image2.jpg")]
image_ids = ["img1", "img2"]
search_engine.add_images(images, image_ids)

# Search for similar images
query_image = Image.open("query.jpg")
results = search_engine.search(query_image, k=5)
```

## Image Clustering

The `ImageClusterer` class helps organize image collections by grouping similar images:

```python
from image_similarity_search import ImageClusterer

# Initialize the clusterer
clusterer = ImageClusterer(
    model_type="clip",  # or "sscd" or "dinov2"
    use_gpu=True,      # Set to False for CPU-only
    threshold=0.1,    # Distance threshold for clustering
    batch_size=32,     # Batch size for processing,
    model_name="openai/clip-vit-large-patch14-336"
)

# Process images and get clusters
clusters = clusterer.process_images("path/to/image/directory")

# Organize clustered images into directories
clusterer.organize_clusters(
    clusters,
    output_directory="path/to/output",
    min_cluster_size=2  # Minimum images to form a cluster
)
```

## Command Line Interface

The package provides a command-line interface for adding images to an index and searching for similar images:

### Adding Images
```bash
python -m image_similarity_search.search_engine add \
    --images image1.jpg image2.jpg \
    --index myindex.faiss \
    --model-type clip \
    --use-gpu
```

Optional arguments:
- `--ids`: Custom IDs for the images
- `--model-name`: Specific model name (required for SSCD)
- `--index-type`: FAISS index type ('l2' or 'ivf')

### Searching Images
```bash
python -m image_similarity_search.search_engine search \
    --query query.jpg \
    --index myindex.faiss \
    --k 5 \
    --model-type clip \
    --use-gpu
```

Optional arguments:
- `--model-name`: Specific model name (required for SSCD)
- `--k`: Number of results to return (default: 5)

## Features

- Multiple operation modes:
  - Similarity Search: Find images similar to a query image
  - Clustering: Group similar images automatically
- Support for multiple embedding models:
  - CLIP (OpenAI)
  - SSCD (Self-Supervised Descriptor for Image Copy Detection)
  - DINOv2 (Facebook AI)
- Multiple index types for different dataset sizes
- Batch processing support
- Hierarchical clustering with customizable thresholds
- Automatic organization of clustered images

## Requirements

- Python >= 3.8
- PyTorch
- FAISS
- Transformers
- Pillow