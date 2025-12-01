from typing import List, Tuple

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


# ===== BASE CLASS =====
class BaseExtractor:
    """Abstract base class for feature extractors."""

    def __init__(self, model_name: str, device):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        self.dim = 0
        self._load_model()

    def _load_model(self):
        """Load model - to be implemented by child classes."""
        raise NotImplementedError

    def extract(
        self, image_paths: List[str], filenames: List[str], batch_size: int = 32
    ) -> Tuple[np.ndarray, List[str]]:
        """Extract features - to be implemented by child classes."""
        raise NotImplementedError


# ===== RESNET EXTRACTOR (PERCEPTUAL) =====
class ResNetExtractor(BaseExtractor):
    """
    Extracts 2048-dim perceptual vectors using ResNet50.
    Pre-trained on ImageNet for robust perceptual features.
    """

    def __init__(self, device):
        super().__init__("ResNet50 (Perceptual)", device)

    def _load_model(self):
        """Load ResNet50 model without final classification layer."""
        print(f"Loading model {self.model_name}...")

        # Load pre-trained ResNet50
        self.model = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1
        )

        # Remove the final FC layer to get feature vectors
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval()

        # Image preprocessing pipeline
        self.processor = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.dim = 2048  # ResNet50 feature dimension

        print(
            f"✓ Successfully loaded {self.model_name} ({self.dim}-dim vectors)"
        )

    def extract(
        self, image_paths: List[str], filenames: List[str], batch_size: int = 32
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract perceptual features from images.

        Args:
            image_paths: List of full image paths
            filenames: List of filenames (for tracking)
            batch_size: Batch size for processing

        Returns:
            Tuple of (features, valid_filenames)
        """
        valid_features_list = []
        valid_filenames_list = []
        tensors = []
        temp_filenames = []

        # ===== PHASE 1: VALIDATE & PREPROCESS =====
        print(f"Validating {len(image_paths)} images...")
        for i, img_path in enumerate(image_paths):
            try:
                # Open and convert to RGB
                img = Image.open(img_path).convert("RGB")

                # Apply preprocessing
                tensor = self.processor(img)
                tensors.append(tensor)
                temp_filenames.append(filenames[i])

            except Exception as e:
                print(
                    f"⚠️ WARNING: Skipping corrupted image (ResNet): {img_path}"
                )
                print(f"   Error: {e}")

        if not tensors:
            print("❌ No valid images found!")
            return np.array([]), []

        print(f"✓ Validated {len(tensors)} images")

        # ===== PHASE 2: BATCH EXTRACTION =====
        print(f"Extracting features in batches of {batch_size}...")

        num_batches = (len(tensors) + batch_size - 1) // batch_size

        for i in range(0, len(tensors), batch_size):
            batch_idx = i // batch_size + 1
            batch_tensors = torch.stack(tensors[i : i + batch_size]).to(
                self.device
            )
            batch_filenames = temp_filenames[i : i + batch_size]

            try:
                with torch.no_grad():
                    features = self.model(batch_tensors)

                # Reshape from (N, 2048, 1, 1) to (N, 2048)
                features_np = (
                    features.cpu().numpy().reshape(len(batch_tensors), -1)
                )

                # L2 Normalization
                norms = np.linalg.norm(features_np, axis=1, keepdims=True)
                norms[norms == 0] = 1.0  # Avoid division by zero
                normalized_features = features_np / norms

                valid_features_list.append(normalized_features)
                valid_filenames_list.extend(batch_filenames)

                print(
                    f"  Batch {batch_idx}/{num_batches}: Processed {len(batch_tensors)} images"
                )

            except Exception as e:
                print(f"❌ ERROR: Processing ResNet batch {i}: {e}")

        if not valid_features_list:
            print("❌ No features extracted!")
            return np.array([]), []

        # ===== PHASE 3: COMBINE RESULTS =====
        all_features = np.vstack(valid_features_list)

        print(
            f"✓ Extracted {all_features.shape[0]} feature vectors of dimension {self.dim}"
        )

        return all_features, valid_filenames_list


# ===== UTILITY FUNCTIONS =====
def compute_similarity_matrix(features: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix.

    Args:
        features: Feature matrix (N, D)

    Returns:
        Similarity matrix (N, N)
    """
    # Assuming features are already L2-normalized
    similarity = np.dot(features, features.T)
    return similarity


def find_most_similar(
    query_features: np.ndarray, database_features: np.ndarray, top_k: int = 5
) -> np.ndarray:
    """
    Find most similar images to query.

    Args:
        query_features: Query feature vector (1, D)
        database_features: Database features (N, D)
        top_k: Number of results to return

    Returns:
        Indices of top_k most similar images
    """
    # Compute similarities
    similarities = np.dot(database_features, query_features.T).squeeze()

    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]

    return top_indices


def batch_extract_with_progress(
    extractor: BaseExtractor,
    image_paths: List[str],
    filenames: List[str],
    batch_size: int = 32,
    callback=None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract features with progress callback.

    Args:
        extractor: Feature extractor instance
        image_paths: List of image paths
        filenames: List of filenames
        batch_size: Batch size
        callback: Callback function(current, total)

    Returns:
        Tuple of (features, valid_filenames)
    """
    features, valid_files = extractor.extract(
        image_paths, filenames, batch_size
    )

    if callback:
        callback(len(features), len(image_paths))

    return features, valid_files


def save_features(
    features: np.ndarray, filenames: List[str], output_path: str
) -> None:
    """
    Save extracted features to disk.

    Args:
        features: Feature matrix
        filenames: Corresponding filenames
        output_path: Output .npz file path
    """
    np.savez_compressed(
        output_path, features=features, filenames=np.array(filenames)
    )
    print(f"✓ Saved {len(features)} features to {output_path}")


def load_features(input_path: str) -> Tuple[np.ndarray, List[str]]:
    """
    Load features from disk.

    Args:
        input_path: Input .npz file path

    Returns:
        Tuple of (features, filenames)
    """
    data = np.load(input_path)
    features = data["features"]
    filenames = data["filenames"].tolist()

    print(f"✓ Loaded {len(features)} features from {input_path}")
    return features, filenames


# ===== TESTING =====
if __name__ == "__main__":
    import os

    print("Testing ResNet Feature Extractor...")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create extractor
    extractor = ResNetExtractor(device)

    # Test with dummy images (if available)
    test_dir = "./test_images"
    if os.path.exists(test_dir):
        image_paths = [
            os.path.join(test_dir, f)
            for f in os.listdir(test_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]

        if image_paths:
            print(f"\nTesting with {len(image_paths)} images from {test_dir}")

            filenames = [os.path.basename(p) for p in image_paths]

            # Extract features
            features, valid_files = extractor.extract(
                image_paths, filenames, batch_size=8
            )

            print("\nResults:")
            print(f"  • Input images: {len(image_paths)}")
            print(f"  • Valid images: {len(valid_files)}")
            print(f"  • Feature shape: {features.shape}")
            print(f"  • Feature dimension: {extractor.dim}")

            # Test similarity
            if len(features) > 1:
                sim_matrix = compute_similarity_matrix(features)
                print("\nSimilarity Matrix:")
                print(f"  • Shape: {sim_matrix.shape}")
                print(f"  • Min similarity: {np.min(sim_matrix):.4f}")
                print(f"  • Max similarity: {np.max(sim_matrix):.4f}")
                print(f"  • Mean similarity: {np.mean(sim_matrix):.4f}")

            # Test save/load
            save_features(features, valid_files, "test_features.npz")
            loaded_features, loaded_files = load_features("test_features.npz")

            assert np.allclose(features, loaded_features)
            assert valid_files == loaded_files
            print("\n✓ Save/Load test passed!")

            # Cleanup
            os.remove("test_features.npz")
        else:
            print(f"No images found in {test_dir}")
    else:
        print(f"Test directory {test_dir} not found")
        print("Creating dummy test...")

        # Create dummy images for testing
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 5 dummy images
            test_paths = []
            test_names = []

            for i in range(5):
                img = Image.new(
                    "RGB", (224, 224), color=(i * 50, i * 30, i * 20)
                )
                path = os.path.join(tmpdir, f"test_{i}.jpg")
                img.save(path)
                test_paths.append(path)
                test_names.append(f"test_{i}.jpg")

            # Extract features
            features, valid_files = extractor.extract(test_paths, test_names)

            print("\nDummy Test Results:")
            print("  • Created: 5 images")
            print(f"  • Extracted: {len(features)} features")
            print(f"  • Feature shape: {features.shape}")

    print("\n✓ ResNet Extractor test complete!")
