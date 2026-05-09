"""File type constants and checker functions for OCR processing."""

# File type constants - following Open/Closed Principle
CONVERTING_IMAGE_EXTS = {"image/png", "image/tiff", "image/webp"}
IMAGE_EXTS = {"image/jpeg"}
DOCX_EXTS = {"application/vnd.openxmlformats-officedocument.wordprocessingml.document"}
PDF_EXTS = {"application/pdf"}
PPTX_EXTS = {
    "application/vnd.openxmlformats-officedocument.presentationml.presentation"
}
COMPRESSED_EXTS = {
    "application/zip",
    "application/x-zip-compressed",
    "application/x-bzip2",
    "application/x-bz2",
    "application/zstd",
    "application/gzip",
    "application/x-gzip",
    "application/x-tar",
    "application/x-7z-compressed",
    "application/x-rar-compressed",
}


# File type checker functions
def is_docx(file_type: str) -> bool:
    """Check if file type is DOCX."""
    return file_type in DOCX_EXTS


def is_pptx(file_type: str) -> bool:
    """Check if file type is PPTX."""
    return file_type in PPTX_EXTS


def is_pdf(file_type: str) -> bool:
    """Check if file type is PDF."""
    return file_type in PDF_EXTS


def is_image(file_type: str) -> bool:
    """Check if file type is an image."""
    return file_type in (IMAGE_EXTS | CONVERTING_IMAGE_EXTS)


def is_ocr_required(file_type: str) -> bool:
    """Check if OCR is required for the file type."""
    return (
        file_type in PDF_EXTS
        or file_type in IMAGE_EXTS
        or file_type in CONVERTING_IMAGE_EXTS
    )


def is_compressed_file(file_type: str) -> bool:
    """Check if file type is a compressed file."""
    return file_type in COMPRESSED_EXTS


def can_process_directly(file_type: str) -> bool:
    """Check if file can be processed directly without OCR."""
    return file_type in (DOCX_EXTS | PPTX_EXTS)
