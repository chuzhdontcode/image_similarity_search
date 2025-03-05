# Image Similarity Search Engine

A flexible image similarity search engine that supports multiple embedding models (CLIP, SSCD, DINOv2) and both CPU and GPU operations using FAISS.

## Installation

For CPU-only usage:
```bash
pip install .
```

For GPU support:
```bash
pip install ".[gpu]"
```

## Usage

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

# Save and load index
search_engine.save_index("index.faiss")
search_engine.load_index("index.faiss")
```

## Features

- Support for multiple embedding models:
  - CLIP (OpenAI)
  - SSCD (Simple Semantic Contrastive Distillation)
  - DINOv2 (Facebook AI)
- GPU acceleration with FAISS-GPU
- Multiple index types for different dataset sizes
- Easy saving and loading of indices
- Batch processing support

## Requirements

- Python >= 3.8
- PyTorch
- FAISS (CPU or GPU version)
- Transformers
- Pillow