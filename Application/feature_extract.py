import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Import directly from transformers
from transformers import CLIPModel, CLIPProcessor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Sử dụng device: {device}")

NUM_TTA_TRANSFORMS = 5  # Using 5 TTA versions: Original, Flip, 2x Rotate, Crop


# --- Dataset with TTA (Quét đệ quy - Remains the same) ---
class ImageTTADataset(Dataset):
    """
    Dataset quét đệ quy VÀ trả về 1 list gồm NUM_TTA_TRANSFORMS ảnh TTA cho mỗi item.
    """

    def __init__(self, root_folder):
        self.root_folder = root_folder
        valid_extensions = (".jpg", ".jpeg", ".png")
        self.filepaths = []
        self.filenames = []

        print(f"Đang quét (recursive) thư mục: {root_folder}...")
        for dirpath, _, filenames_in_dir in os.walk(root_folder):
            for f in filenames_in_dir:
                if f.lower().endswith(valid_extensions):
                    full_path = os.path.join(dirpath, f)
                    self.filepaths.append(full_path)
                    relative_path = os.path.relpath(full_path, self.root_folder)
                    self.filenames.append(relative_path.replace("\\", "/"))
        print(f"Đã tìm thấy {len(self.filepaths)} ảnh.")

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img_path = self.filepaths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            w, h = image.size
            tta_transforms = [
                lambda img: img,
                lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),
                lambda img: img.rotate(15, expand=False, fillcolor="black"),
                lambda img: img.rotate(-15, expand=False, fillcolor="black"),
                lambda img: img.crop(
                    (w * 0.3, h * 0.3, w * 0.7, h * 0.7)
                ).resize((w, h)),
            ]
            tta_pil_images = [transform(image) for transform in tta_transforms]
            return tta_pil_images, self.filenames[idx]
        except Exception as e:
            print(f"⚠️ Lỗi đọc ảnh {img_path}: {e}")
            return None, self.filenames[idx]


def collate_fn_skip_errors(batch):
    """Bỏ qua các ảnh bị lỗi (None) trong batch."""
    batch = [
        (img_list, name) for img_list, name in batch if img_list is not None
    ]
    if not batch:
        return None, None
    images, names = zip(*batch)
    return list(images), list(names)


# --- CORRECTED CLIP Encoder Class (Using transformers directly) ---
class CLIPEncoderDirect:
    """
    Wrapper for CLIP using transformers library directly.
    """

    def __init__(self, model_name, device="cpu"):
        self.device = device
        self.model_name = model_name

        print(f"Loading CLIP processor for {model_name}...")
        try:
            # Load the specific CLIP processor
            self.processor = CLIPProcessor.from_pretrained(model_name)
            print("✅ Processor loaded successfully.")
        except Exception as e:
            print(f"❌ FAILED TO LOAD CLIP PROCESSOR!")
            raise e

        print(f"Loading CLIP model {model_name}...")
        try:
            # Load the specific CLIP model
            self.model = (
                CLIPModel.from_pretrained(model_name).to(self.device).eval()
            )
            # Get the embedding dimension (CLIP uses projection_dim)
            self.dimension = self.model.config.projection_dim
            print(
                f"✅ Model loaded successfully. Dimension: {self.dimension}"
            )  # L-14 is 768
        except Exception as e:
            print(f"❌ FAILED TO LOAD CLIP MODEL!")
            raise e

        print(f"✅ CLIP (model + processor) loaded to {self.device}")

    def get_embedding_dimension(self):
        return self.dimension

    def encode(
        self,
        images_list,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
    ):
        """
        Encodes a list of PIL images into embeddings using the vision tower.
        """
        all_embeddings = []
        # Process in batches
        for start_index in tqdm(
            range(0, len(images_list), batch_size),
            desc="Encoding Images",
            disable=not show_progress_bar,
        ):
            images_batch = images_list[start_index : start_index + batch_size]

            # Preprocess images using CLIPProcessor
            inputs = self.processor(
                text=None,  # We only care about images
                images=images_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)

            # Get image features from the model
            with torch.no_grad():
                # We only need the image_embeds
                image_features = self.model.get_image_features(
                    **inputs
                )  # Shape: (batch, projection_dim)

            # Embeddings are already L2 normalized by CLIP model's projection
            embeddings = image_features

            if convert_to_numpy:
                all_embeddings.append(embeddings.cpu().numpy())
            else:
                all_embeddings.append(embeddings)

        # Concatenate results from all batches
        if convert_to_numpy:
            return np.concatenate(all_embeddings)
        else:
            return torch.cat(all_embeddings)


# --- Main Feature Extraction Function (Updated to use CLIPEncoderDirect) ---
def extract_features(
    img_dir,
    feature_file,
    name_file,
    model_name="openai/clip-vit-large-patch14",
    batch_size=32,
):
    """
    Trích xuất đặc trưng CLIP siêu ổn định (Stable Features) bằng TTA.
    Uses direct transformers loading.
    """
    print(f"Loading CLIP model ({model_name}) using direct transformers...")
    # Instantiate the direct encoder
    try:
        model = CLIPEncoderDirect(model_name, device=device)
        dimension = model.get_embedding_dimension()  # Get dimension correctly
    except Exception as e:
        print("Failed during model initialization.")
        return np.array([]), np.array([])

    print(f"Creating DataLoader (TTA enabled) for directory: {img_dir}")
    dataset = ImageTTADataset(img_dir)

    if len(dataset) == 0:
        print(f"❌ No images found in directory: {img_dir}")
        return np.array([]), np.array([])

    loader = DataLoader(
        dataset,
        batch_size=batch_size,  # This is the batch size for TTA groups
        shuffle=False,
        num_workers=0,  # Use 0 for stability
        collate_fn=collate_fn_skip_errors,
    )

    all_features_aggregated = []  # Stores the final, averaged TTA features
    all_original_names = []  # Stores the corresponding filenames

    print("Starting feature extraction with TTA...")
    # Loop through batches provided by DataLoader
    for batch_tta_imgs, batch_names in tqdm(loader, desc="Extracting TTA-CLIP"):
        if batch_tta_imgs is None:  # Skip errored batches
            continue

        # batch_tta_imgs is a list of lists: [[5 PILs], [5 PILs], ...] (size = batch_size)

        # 1. Flatten the batch for model processing
        flat_pil_images = [
            img for tta_list in batch_tta_imgs for img in tta_list
        ]

        # 2. Encode all flattened images using the direct encoder's .encode
        embeddings_flat = model.encode(
            flat_pil_images,
            batch_size=len(
                flat_pil_images
            ),  # Process all TTA images of the batch
            show_progress_bar=False,  # tqdm is already running outside
            convert_to_numpy=True,
        )  # Shape: (batch_size * NUM_TTA, embedding_dim)

        # 3. Reshape back to group TTA embeddings
        embeddings_grouped = embeddings_flat.reshape(
            len(batch_names), NUM_TTA_TRANSFORMS, -1
        )

        # 4. Aggregate TTA embeddings (using mean)
        final_aggregated_features = np.mean(embeddings_grouped, axis=1)

        # 5. Optional but recommended: Re-normalize after averaging
        norms = np.linalg.norm(final_aggregated_features, axis=1, keepdims=True)
        final_aggregated_features = final_aggregated_features / (norms + 1e-10)

        all_features_aggregated.extend(final_aggregated_features)
        all_original_names.extend(batch_names)

    # Convert final lists to numpy arrays
    all_features_aggregated = np.array(all_features_aggregated)
    all_original_names = np.array(all_original_names)

    # 6. Save the results
    try:
        np.save(feature_file, all_features_aggregated)
        np.save(name_file, all_original_names)
        print(
            f"🎉 Successfully saved {len(all_original_names)} aggregated TTA-CLIP features to:"
        )
        print(f"   Features: {feature_file}")
        print(f"   Filenames: {name_file}")
    except Exception as e:
        print(f"❌ Error saving numpy files: {e}")

    return all_features_aggregated, all_original_names


# --- Main execution block ---
if __name__ == "__main__":

    CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"  # 768 dimensions
    # CLIP_MODEL_NAME = "openai/clip-vit-base-patch32" # 512 dimensions

    model_short_name = CLIP_MODEL_NAME.split("/")[-1].replace("-", "_")
    FEATURE_FILE_OUT = "features.npy"
    NAME_FILE_OUT = "names.npy"
    DATASET_DIR = "dataset"

    if os.path.exists(FEATURE_FILE_OUT):
        os.remove(FEATURE_FILE_OUT)
    if os.path.exists(NAME_FILE_OUT):
        os.remove(NAME_FILE_OUT)

    extract_features(
        img_dir=DATASET_DIR,
        feature_file=FEATURE_FILE_OUT,
        name_file=NAME_FILE_OUT,
        model_name=CLIP_MODEL_NAME,
        batch_size=16,  # Adjust based on memory
    )
