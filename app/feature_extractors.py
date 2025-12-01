import os
from typing import Callable, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

# Check for Kornia
try:
    import kornia.augmentation as K
    import kornia.filters as KF
except ImportError:
    raise ImportError("Please install kornia: pip install kornia")


# ===== BASE CLASS =====
class BaseExtractor:
    """Abstract base class for feature extractors."""

    def __init__(self, model_name: str, device: str = None):
        self.model_name = model_name
        self.device = torch.device(
            device
            if device
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = None
        self.dim = 0
        self._load_model()

    def _load_model(self):
        """Load model - to be implemented by child classes."""
        raise NotImplementedError

    def extract(
        self, image_paths: List[str], filenames: List[str], batch_size: int = 24
    ) -> Tuple[np.ndarray, List[str]]:
        """Extract features - to be implemented by child classes."""
        raise NotImplementedError


# ===== ROBUST RESNET EXTRACTOR (KORNIA AUGMENTED) =====
class ResNetExtractor(BaseExtractor):
    """
    Extracts robust 2048-dim vectors using ResNet50 + Kornia Augmentations.
    Generates multiple variants of a single image, extracts features for all,
    and returns the MEAN vector.
    """

    def __init__(self, device: str = None):
        super().__init__("ResNet50_Mean_Augmented", device)

        # Define Normalization (ImageNet stats)
        self.normalize = K.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]).to(self.device),
            std=torch.tensor([0.229, 0.224, 0.225]).to(self.device),
        )

        # Initialize the Augmentation Pipeline
        self.augments = self._build_augmentations()

    def _load_model(self):
        print(f"Loading {self.model_name} on {self.device}...")

        # Load ResNet50 with Default weights
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Remove FC layer
        self.model = nn.Sequential(*list(resnet.children())[:-1])
        self.model.to(self.device)
        self.model.eval()

        self.dim = 2048
        print(f"✓ Model Loaded. Feature dimension: {self.dim}")

    def _build_augmentations(self) -> List[Union[nn.Module, Callable]]:
        """Defines the list of Kornia augmentations."""
        augs = [
            # --- Fixed Rotations ---
            K.RandomRotation(degrees=(0.0, 0.0), p=1.0),  # Original
            K.RandomRotation(degrees=(90.0, 90.0), p=1.0),
            K.RandomRotation(degrees=(180.0, 180.0), p=1.0),
            K.RandomRotation(degrees=(270.0, 270.0), p=1.0),
            K.RandomRotation(degrees=(15.0, 15.0), p=1.0),
            K.RandomRotation(degrees=(-15.0, -15.0), p=1.0),
            K.RandomRotation(degrees=(45.0, 45.0), p=1.0),
            K.RandomRotation(degrees=(-45.0, -45.0), p=1.0),
            # --- Flips ---
            K.RandomHorizontalFlip(p=1.0),
            K.RandomVerticalFlip(p=1.0),
            # --- Crops ---
            K.RandomResizedCrop(
                size=(224, 224), scale=(0.8, 0.8), ratio=(1.0, 1.0), p=1.0
            ),
            K.RandomResizedCrop(
                size=(224, 224), scale=(0.9, 0.9), ratio=(1.0, 1.0), p=1.0
            ),
            # --- Color Jitter ---
            K.ColorJitter(brightness=0.5, contrast=0.0, p=1.0),
            K.ColorJitter(brightness=1.5, contrast=0.0, p=1.0),
            K.ColorJitter(brightness=0.0, contrast=0.7, p=1.0),
            K.ColorJitter(brightness=0.0, contrast=1.5, p=1.0),
            # --- Sharpness ---
            K.RandomSharpness(sharpness=0.3, p=1.0),
            K.RandomSharpness(sharpness=2.0, p=1.0),
            # --- Blur / Filters (Lambdas) ---
            lambda x: KF.gaussian_blur2d(x, (3, 3), (2.0, 2.0)),
            lambda x: KF.median_blur(x, (3, 3)),
            lambda x: KF.box_blur(x, (3, 3)),
            lambda x: KF.gaussian_blur2d(x, (3, 3), (0.5, 0.5)),
            lambda x: KF.gaussian_blur2d(x, (5, 5), (1.0, 1.0)),
            # --- Noise ---
            lambda x: x + 0.02 * torch.randn_like(x),
            lambda x: x + 0.05 * torch.randn_like(x),
        ]
        return augs

    def extract(
        self, image_paths: List[str], filenames: List[str], batch_size: int = 24
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extracts features by creating variants of each image, running inference,
        and calculating the mean feature vector.

        Args:
            image_paths: List of full paths to images.
            filenames: List of filenames (IDs).
            batch_size: Max number of augmentations to process in one forward pass (VRAM limit).
        """
        valid_features = []
        valid_names = []

        print(f"Starting Robust Extraction on {len(image_paths)} images...")
        print(f"Augmentations per image: {len(self.augments)}")

        # Basic pre-processing to get tensor onto GPU
        to_tensor = transforms.ToTensor()
        resize_base = transforms.Resize((224, 224))

        for i, (img_path, fname) in enumerate(
            tqdm(
                zip(image_paths, filenames),
                total=len(image_paths),
                desc="Processing",
            )
        ):
            try:
                # 1. Load Image -> Grayscale -> RGB (Focus on structure)
                img = Image.open(img_path).convert("L").convert("RGB")

                # 2. Convert to Tensor (1, 3, H, W)
                img_tensor = to_tensor(img).unsqueeze(0).to(self.device)
                img_tensor = resize_base(img_tensor)

                # 3. Create Batch of Augmentations
                aug_tensors = []
                for aug in self.augments:
                    # Apply augmentation (works for both Kornia Modules and Lambdas)
                    out = aug(img_tensor)
                    aug_tensors.append(out)

                # Stack into a batch: (N_Augments, 3, 224, 224)
                full_aug_batch = torch.cat(aug_tensors, dim=0)
                full_aug_batch = self.normalize(full_aug_batch)

                # 4. Feed to ResNet (in sub-batches if N_Augments > batch_size)
                feature_accum = []

                with torch.no_grad():
                    total_augs = len(full_aug_batch)
                    for j in range(0, total_augs, batch_size):
                        sub_batch = full_aug_batch[j : j + batch_size]

                        # Forward pass
                        feats = self.model(sub_batch)

                        # Flatten: (Batch, 2048, 1, 1) -> (Batch, 2048)
                        feats = feats.view(feats.size(0), -1)
                        feature_accum.append(feats)

                # 5. Compute Mean Feature
                all_feats = torch.cat(
                    feature_accum, dim=0
                )  # (N_Augments, 2048)
                mean_feat = all_feats.mean(dim=0).cpu().numpy()  # (2048,)

                # 6. L2 Normalize result
                norm = np.linalg.norm(mean_feat)
                if norm > 0:
                    mean_feat = mean_feat / norm

                valid_features.append(mean_feat)
                valid_names.append(fname)

            except Exception as e:
                print(f"\n❌ Error processing {fname}: {e}")
                continue

        if not valid_features:
            return np.array([]), []

        return np.array(valid_features), valid_names


# ===== UTILITY FUNCTIONS =====


def save_features(
    features: np.ndarray, filenames: List[str], output_path: str
) -> None:
    """Save extracted features to .npz file."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    np.savez_compressed(
        output_path, features=features, filenames=np.array(filenames)
    )
    print(f"✓ Saved {len(features)} features to {output_path}")


def load_features(input_path: str) -> Tuple[np.ndarray, List[str]]:
    """Load features from .npz file."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} does not exist.")

    data = np.load(input_path)
    features = data["features"]
    filenames = data["filenames"].tolist()

    print(f"✓ Loaded {len(features)} features from {input_path}")
    return features, filenames
