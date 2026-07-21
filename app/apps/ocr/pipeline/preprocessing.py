"""Image preprocessing for OCR quality improvement."""

import logging
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Apply preprocessing steps to improve OCR quality."""

    def __init__(
        self,
        enable_deskew: bool = True,
        enable_contrast: bool = True,
        enable_denoise: bool = False,
    ) -> None:
        self.enable_deskew = enable_deskew
        self.enable_contrast = enable_contrast
        self.enable_denoise = enable_denoise

    def process(self, image: Image.Image) -> Image.Image:
        """Apply preprocessing to a PIL Image, return processed PIL Image."""
        img = self._pil_to_cv2(image)
        steps: list[str] = []

        if self.enable_deskew:
            angle, img = self._deskew(img)
            if abs(angle) > 0.5:
                steps.append(f"deskew({angle:.1f})")

        if self.enable_contrast and self._needs_contrast(img):
            img = self._enhance_contrast(img)
            steps.append("contrast")

        if self.enable_denoise and self._needs_denoise(img):
            img = self._denoise(img)
            steps.append("denoise")

        if steps:
            logger.debug("Preprocessing applied: %s", ", ".join(steps))

        return self._cv2_to_pil(img)

    def process_bytes(self, image_bytes: BytesIO) -> Image.Image:
        """Process image from BytesIO."""
        image = Image.open(image_bytes)
        return self.process(image)

    @staticmethod
    def _pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
        return cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
        return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

    @staticmethod
    def _deskew(img: np.ndarray) -> tuple[float, np.ndarray]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) < 10:
            return 0.0, img
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        if abs(angle) < 0.5:
            return 0.0, img
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            img, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
        return float(angle), rotated

    @staticmethod
    def _needs_contrast(img: np.ndarray) -> bool:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return float(np.std(gray)) < 40

    @staticmethod
    def _enhance_contrast(img: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lightness, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lightness = clahe.apply(lightness)
        enhanced = cv2.merge([lightness, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    @staticmethod
    def _needs_denoise(img: np.ndarray) -> bool:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var() < 100

    @staticmethod
    def _denoise(img: np.ndarray) -> np.ndarray:
        return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
