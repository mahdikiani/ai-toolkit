"""Image conversion utilities for PIL Image and BytesIO objects."""

from io import BytesIO

from PIL import Image


def convert_to_jpg(image: Image.Image | BytesIO) -> Image.Image:
    """
    Convert an image to RGB JPEG format.

    Args:
        image: PIL Image or BytesIO containing image data.

    Returns:
        RGB-mode PIL Image.
    """
    if isinstance(image, BytesIO):
        image = Image.open(image)
    return image.convert("RGB")


def convert_to_jpg_bytes(image: Image.Image | BytesIO) -> BytesIO:
    """
    Convert an image to JPEG bytes.

    Args:
        image: PIL Image or BytesIO containing image data.

    Returns:
        BytesIO object containing JPEG-encoded image data.
    """
    if isinstance(image, BytesIO):
        image = Image.open(image)
    output = BytesIO()
    image.save(output, format="jpeg")
    output.seek(0)
    return output
