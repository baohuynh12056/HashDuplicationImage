from typing import Dict, List, Tuple

import cv2
import numpy as np


class ImageQualityScorer:
    """
    Đánh giá chất lượng ảnh với penalizations cho:
    - Ảnh grayscale
    - Ảnh bị crop/rotate với fill đen
    - Ảnh blur
    - Ảnh có biến đổi
    """

    def __init__(self):
        self.weights = {
            "resolution": 0.20,
            "sharpness": 0.30,
            "exposure": 0.15,
            "contrast": 0.10,
            "noise": 0.05,
            "color_richness": 0.15,
            "edge_integrity": 0.05,
        }

    def calculate_resolution_score(self, image: np.ndarray) -> float:
        """Điểm dựa trên độ phân giải."""
        height, width = image.shape[:2]
        total_pixels = height * width
        mp = total_pixels / 1_000_000
        score = min(100, (mp / 12) * 100)
        return score

    def calculate_sharpness_score(self, image: np.ndarray) -> float:
        """Điểm dựa trên độ sắc nét (Laplacian variance)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        score = min(100, (laplacian_var / 500) * 100)
        return score

    def calculate_exposure_score(self, image: np.ndarray) -> float:
        """Điểm dựa trên độ sáng."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        optimal = 127
        deviation = abs(mean_brightness - optimal)
        score = max(0, 100 - (deviation / optimal) * 100)
        return score

    def calculate_contrast_score(self, image: np.ndarray) -> float:
        """Điểm dựa trên độ tương phản."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        std_dev = np.std(gray)
        score = min(100, (std_dev / 70) * 100)
        return score

    def calculate_noise_score(self, image: np.ndarray) -> float:
        """Điểm dựa trên độ nhiễu."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size

        if 0.05 <= edge_ratio <= 0.15:
            score = 100
        elif edge_ratio < 0.05:
            score = (edge_ratio / 0.05) * 100
        else:
            score = max(0, 100 - ((edge_ratio - 0.15) / 0.15) * 100)

        return score

    def calculate_color_richness_score(self, image: np.ndarray) -> float:
        """
        ✨ Điểm màu sắc - Phạt ảnh grayscale hoặc desaturated.
        Ảnh màu phong phú = 100, grayscale = 0
        """
        b, g, r = cv2.split(image)

        std_diff_rg = np.std(r - g)
        std_diff_rb = np.std(r - b)
        std_diff_gb = np.std(g - b)

        color_variance = (std_diff_rg + std_diff_rb + std_diff_gb) / 3

        if color_variance < 5:
            score = 0
        elif color_variance > 30:
            score = 100
        else:
            score = (color_variance / 30) * 100

        return score

    def calculate_edge_integrity_score(self, image: np.ndarray) -> float:
        """
        ✨ MỚI: Kiểm tra ảnh bị crop/rotate với fill đen ở góc.
        Phát hiện vùng đen ở 4 góc ảnh.
        """
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        corner_size = int(min(height, width) * 0.1)

        corners = [
            gray[0:corner_size, 0:corner_size],
            gray[0:corner_size, -corner_size:],
            gray[-corner_size:, 0:corner_size],
            gray[-corner_size:, -corner_size:],
        ]

        black_corners = sum(1 for corner in corners if np.mean(corner) < 30)

        if black_corners == 0:
            score = 100
        elif black_corners == 1:
            score = 80
        elif black_corners == 2:
            score = 40
        else:
            score = 0

        return score

    def detect_filename_modifications(self, filename: str) -> float:
        """
        ✨ MỚI: Phạt ảnh có dấu hiệu modification trong tên file.
        """
        filename_lower = filename.lower()

        penalties = {
            "rotate": -20,
            "flip": -15,
            "grayscale": -25,
            "blur": -30,
            "noise": -20,
            "darker": -10,
            "brighter": -10,
            "desaturate": -20,
            "hue": -15,
            "saturation": -15,
        }

        penalty = 0
        for keyword, value in penalties.items():
            if keyword in filename_lower:
                penalty += value

        if "original" in filename_lower:
            penalty += 30

        return max(-100, min(30, penalty))

    def score_image(self, image_path: str) -> Dict[str, float]:
        """
        Tính toán tổng điểm chất lượng cho một ảnh.
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Cannot read image: {image_path}")

            scores = {
                "resolution": self.calculate_resolution_score(image),
                "sharpness": self.calculate_sharpness_score(image),
                "exposure": self.calculate_exposure_score(image),
                "contrast": self.calculate_contrast_score(image),
                "noise": self.calculate_noise_score(image),
                "color_richness": self.calculate_color_richness_score(image),
                "edge_integrity": self.calculate_edge_integrity_score(image),
            }
            total_score = sum(
                scores[metric] * self.weights[metric] for metric in scores
            )

            import os

            filename = os.path.basename(image_path)
            filename_penalty = self.detect_filename_modifications(filename)
            total_score += filename_penalty

            total_score = max(0, min(100, total_score))

            scores["total"] = round(total_score, 2)
            scores["filename_bonus"] = filename_penalty

            for key in scores:
                if key not in ["total", "filename_bonus"]:
                    scores[key] = round(scores[key], 2)

            return scores

        except Exception as e:
            print(f"Error scoring image {image_path}: {e}")
            return {
                "resolution": 0,
                "sharpness": 0,
                "exposure": 0,
                "contrast": 0,
                "noise": 0,
                "color_richness": 0,
                "edge_integrity": 0,
                "total": 0,
                "filename_bonus": 0,
                "error": str(e),
            }

    def score_cluster(
        self, image_paths: List[str]
    ) -> List[Tuple[str, Dict[str, float]]]:
        """
        Chấm điểm tất cả ảnh trong một cluster và sắp xếp theo chất lượng.
        """
        results = []

        for img_path in image_paths:
            scores = self.score_image(img_path)
            results.append((img_path, scores))

        results.sort(key=lambda x: x[1]["total"], reverse=True)

        return results

    def get_best_image(
        self, image_paths: List[str]
    ) -> Tuple[str, Dict[str, float]]:
        """
        Tìm ảnh tốt nhất trong cluster.
        """
        scored_images = self.score_cluster(image_paths)
        return scored_images[0] if scored_images else (None, {})

    def get_quality_label(self, score: float) -> str:
        """Convert điểm số thành nhãn dễ hiểu."""
        if score >= 80:
            return "Excellent"
        elif score >= 60:
            return "Good"
        elif score >= 40:
            return "Fair"
        elif score >= 20:
            return "Poor"
        else:
            return "Very Poor"

    def get_quality_color(self, score: float) -> str:
        """Trả về màu sắc cho UI dựa trên điểm."""
        if score >= 80:
            return "#10b981"
        elif score >= 60:
            return "#3b82f6"
        elif score >= 40:
            return "#f59e0b"
        elif score >= 20:
            return "#ef4444"
        else:
            return "#991b1b"
