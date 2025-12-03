import logging
import numpy as np
import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)

# Global state for lazy-loading PaddleOCR
_paddle_available = None
_paddle_ocr = None


def _init_paddleocr():
    """
    Lazy-load PaddleOCR. This avoids import errors or crashes on Windows
    if PaddlePaddle or PaddleOCR are not installed or not compatible.
    """
    global _paddle_available, _paddle_ocr

    # If already initialized
    if _paddle_available is not None:
        return _paddle_available

    try:
        from paddleocr import PaddleOCR
        _paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')
        _paddle_available = True
        logger.info("PaddleOCR initialized successfully.")
    except Exception as e:
        _paddle_available = False
        _paddle_ocr = None

    return _paddle_available


def ocr_with_paddle(image: Image.Image) -> str:
    """
    Run PaddleOCR on an image. Raises exception if PaddleOCR is unavailable.
    """
    ok = _init_paddleocr()
    if not ok:
        raise RuntimeError("PaddleOCR is not available.")

    arr = np.array(image)
    result = _paddle_ocr.ocr(arr)
    lines = []

    for block in result:
        for line in block:
            text = line[1][0]
            lines.append(text)

    return "\n".join(lines)


def ocr_tesseract(image: Image.Image) -> str:
    """
    Run Tesseract OCR as fallback.
    Always works on Windows if Tesseract is installed.
    """
    try:
        return pytesseract.image_to_string(image)
    except Exception as e:
        logger.exception("❌ Tesseract OCR failed: %s", e)
        return ""


def ocr_try_best(image: Image.Image) -> str:
    """
    High-level OCR function:
    - Try PaddleOCR first (if available)
    - Fall back to Tesseract
    """
    try:
        return ocr_with_paddle(image)
    except Exception:
        logger.info("⚠️ PaddleOCR not available or failed — using Tesseract instead.")
        return ocr_tesseract(image)
