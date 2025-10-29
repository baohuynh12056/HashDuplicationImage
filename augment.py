import os
from io import BytesIO

import numpy as np
import requests
from PIL import Image, ImageEnhance, ImageFilter


def add_gaussian_noise(image, mean=0, var=0.1):
    """
    Adds Gaussian noise to a PIL image.
    """
    img_np = np.array(image) / 255.0  # Convert image to [0, 1] float range
    # Create noise
    sigma = var**0.5
    gaussian = np.random.normal(mean, sigma, img_np.shape)
    # Add noise and clip values back to [0, 1] range
    noisy_img_np = np.clip(img_np + gaussian, 0, 1)
    # Convert back to [0, 255] and uint8 type
    noisy_img_np = (noisy_img_np * 255).astype(np.uint8)
    # Convert back to a PIL image
    return Image.fromarray(noisy_img_np)


def adjust_hue(image, factor=0.5):
    """
    Adjusts the hue of a PIL image.
    factor: 0.0-1.0 to shift hue. 0.0 is original, 0.5 shifts by 180 degrees.
    """
    # PIL doesn't have a direct hue method, must convert to HSV
    hsv_img = image.convert("HSV")
    hsv_np = np.array(hsv_img)

    width, height = image.size

    # The HUE channel is hsv_np[:,:,0]. In PIL, values are 0-255.
    # To shift by 180 deg (factor=0.5), we add 128 (0.5 * 256)
    hue_shift = int(256 * factor)
    # 1. Cast the hue channel to a larger integer type (e.g., int32) for calculation
    hue_channel = hsv_np[:, :, 0].astype(np.int32)

    # 2. Perform addition and modulo. This is now safe as int32 can hold 328
    hue_channel = (hue_channel + hue_shift) % 256
    # Then take modulo 256 to wrap around the 0-255 range
    hsv_np[:, :, 0] = hue_channel.astype(np.uint8)

    raw_data = hsv_np.tobytes()

    # Convert back to a PIL image and then to RGB format
    return Image.frombytes("HSV", (width, height), raw_data).convert("RGB")


def create_augmented_dataset(url, label, main_dir="dataset"):
    """
    Downloads an image from a URL, creates 4 augmented versions,
    and saves all 5 images into a labeled subdirectory (e.g., dataset/Dog).
    """
    try:
        # 1. Create the subdirectory based on the label
        # os.path.join ensures the path is correct for any OS
        output_dir = os.path.join(main_dir, label)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"✅ Created directory: {output_dir}")

        # 2. Download and open the image
        print(f"⏳ Downloading image for label '{label}'...")
        response = requests.get(url)
        response.raise_for_status()  # Raise an error if the URL is invalid

        original_img = Image.open(BytesIO(response.content)).convert("RGB")

        # Get a base filename for naming the images
        base_filename = label.lower()  # e.g., 'dog'

        # 3. Save the original and augmented versions
        print("⏳ Processing and saving images...")

        # (1) Original
        original_img.save(
            os.path.join(output_dir, f"{base_filename}_01_original.jpg"), "JPEG"
        )

        # (2) Rotate Left
        rotate_left_img = original_img.rotate(
            5, expand=False, fillcolor="black"
        )
        rotate_left_img.save(
            os.path.join(output_dir, f"{base_filename}_02_rotate_left.jpg"),
            "JPEG",
        )

        # (3) Rotate Right
        rotate_right_img = original_img.rotate(
            -5, expand=False, fillcolor="black"
        )
        rotate_right_img.save(
            os.path.join(output_dir, f"{base_filename}_03_rotate_right.jpg"),
            "JPEG",
        )

        # (4) Crop / Zoom
        width, height = original_img.size
        zoomed_img = original_img.crop(
            (width * 0.15, height * 0.15, width * 0.85, height * 0.85)
        )
        # Resize
        zoomed_img = zoomed_img.resize((width, height))
        zoomed_img.save(
            os.path.join(output_dir, f"{base_filename}_04_crop_zoom.jpg"),
            "JPEG",
        )

        # (5) Horizontal Flip
        flip_img = original_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        flip_img.save(
            os.path.join(output_dir, f"{base_filename}_05_flip_horizontal.jpg"),
            "JPEG",
        )

        """# (6) Blurred
        blurred_img = original_img.filter(ImageFilter.GaussianBlur(radius=5))
        blurred_img.save(
            os.path.join(output_dir, f"{base_filename}_06_blurred.jpg"), "JPEG"
        )

        # (7) Darker
        dark_enhancer = ImageEnhance.Brightness(original_img)
        darker_img = dark_enhancer.enhance(0.8)
        darker_img.save(
            os.path.join(output_dir, f"{base_filename}_07_darker.jpg"), "JPEG"
        )

        # (8) Desaturation (Giữ lại - màu xỉn)
        desat_enhancer = ImageEnhance.Color(original_img)
        desaturated_img = desat_enhancer.enhance(0.8)
        desaturated_img.save(
            os.path.join(output_dir, f"{base_filename}_08_desaturated.jpg"),
            "JPEG",
        )

        # (9) Saturation
        sat_enhancer = ImageEnhance.Color(original_img)
        saturated_img = sat_enhancer.enhance(1.5)
        saturated_img.save(
            os.path.join(output_dir, f"{base_filename}_09_saturation.jpg"),
            "JPEG",
        )

        # (10) Add Noise
        noisy_img = add_gaussian_noise(original_img, mean=0, var=0.05)
        noisy_img.save(
            os.path.join(output_dir, f"{base_filename}_10_noise.jpg"), "JPEG"
        )

        print(f"🎉 Success! Saved 10 images to '{output_dir}'.")
        """

    except requests.exceptions.RequestException as e:
        print(f"❌ Error downloading image: {e}")
    except Exception as e:
        print(f"❌ An error occurred: {e}")


# --- START EXECUTING HERE ---
if __name__ == "__main__":

    image_list = [
        (
            "Dog",
            "https://images.pexels.com/photos/220938/pexels-photo-220938.jpeg",
        ),
        (
            "Cat",
            "https://images.pexels.com/photos/45201/kitty-cat-kitten-pet-45201.jpeg",
        ),
        (
            "Elephant",
            "https://images.pexels.com/photos/1054655/pexels-photo-1054655.jpeg",
        ),
        (
            "Parrot",
            "https://images.pexels.com/photos/1059823/pexels-photo-1059823.jpeg",
        ),
        (
            "Dolphin",
            "https://images.pexels.com/photos/64219/dolphin-marine-mammals-water-sea-64219.jpeg",
        ),
        (
            "Monkey",
            "https://images.pexels.com/photos/1429812/pexels-photo-1429812.jpeg",
        ),
        (
            "Rabbit",
            "https://images.pexels.com/photos/2285996/pexels-photo-2285996.jpeg",
        ),
        (
            "Lion",
            "https://images.pexels.com/photos/46795/lion-big-cat-predator-safari-46795.jpeg",
        ),
        (
            "Fish",
            "https://images.pexels.com/photos/1145274/pexels-photo-1145274.jpeg",
        ),
        (
            "Goat",
            "https://images.pexels.com/photos/144240/goat-lamb-little-grass-144240.jpeg",
        ),
    ]

    for label, url in image_list:
        create_augmented_dataset(url=url, label=label)
        print("-" * 20)
