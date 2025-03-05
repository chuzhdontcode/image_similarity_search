import argparse
from image_similarity_search.clustering import ImageClusterer


def main():
    parser = argparse.ArgumentParser(description="Cluster similar images using embeddings")
    parser.add_argument(
        "--image_directory",
        type=str,
        required=True,
        help="Path to the directory containing images to cluster",
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        required=True,
        help="Path to output the clustered images",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="clip",
        choices=["clip", "sscd", "dinov2"],
        help="Model to use for embedding generation",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.22,
        help="Threshold for hierarchical clustering",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=192,
        help="Batch size for processing images",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Use GPU for embedding generation if available",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Specific model name (required for SSCD)",
    )
    parser.add_argument(
        "--min_cluster_size",
        type=int,
        default=2,
        help="Minimum number of images to form a cluster",
    )
    parser.add_argument(
        "--normalize_features",
        action="store_true",
        default=True,
        help="Whether to normalize feature vectors",
    )

    args = parser.parse_args()

    clusterer = ImageClusterer(
        model_type=args.model_type,
        use_gpu=args.use_gpu,
        threshold=args.threshold,
        batch_size=args.batch_size,
        model_name=args.model_name,
        normalize_features=args.normalize_features,
    )

    print(f"Processing images in {args.image_directory}...")
    clusters = clusterer.process_images(args.image_directory)

    print(f"Organizing clusters in {args.output_directory}...")
    clusterer.organize_clusters(
        clusters,
        args.output_directory,
        args.min_cluster_size,
    )

    print("Done!")


if __name__ == "__main__":
    main()
