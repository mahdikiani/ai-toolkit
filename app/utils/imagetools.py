"""Provide module functionality."""
from io import BytesIO

from PIL import Image


def convert_to_jpg(image: Image.Image | BytesIO) -> Image.Image:
    """Run convert to jpg."""
    if isinstance(image, BytesIO):
        image = Image.open(image)
    return image.convert("RGB")


def convert_to_jpg_bytes(image: Image.Image | BytesIO) -> BytesIO:
    """Run convert to jpg bytes."""
    if isinstance(image, BytesIO):
        image = Image.open(image)
    output = BytesIO()
    image.save(output, format="jpeg")
    output.seek(0)
    return output
