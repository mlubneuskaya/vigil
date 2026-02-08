"""
Mask Utilities Module

Provides functions for mask encoding/decoding (RLE format) and conversions.
Uses pycocotools for COCO-compatible RLE encoding.
"""

from typing import Dict, List, Tuple, Any
import numpy as np


def encode_rle(mask: np.ndarray) -> Dict[str, Any]:
    """
    Encode a binary mask to RLE (Run-Length Encoding) format.

    Uses pycocotools for COCO-compatible encoding.

    Args:
        mask: 2D binary numpy array (H, W) with bool or uint8 values

    Returns:
        Dict with 'counts' (RLE string) and 'size' [height, width]
    """
    from pycocotools import mask as mask_utils

    # Ensure mask is in correct format (Fortran order, uint8)
    if mask.dtype == bool:
        mask = mask.astype(np.uint8)

    # pycocotools expects Fortran-ordered array
    mask_fortran = np.asfortranarray(mask)

    # Encode to RLE
    rle = mask_utils.encode(mask_fortran)

    # Convert bytes to string for JSON serialization
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("utf-8")

    return {
        "counts": rle["counts"],
        "size": list(rle["size"]),  # [height, width]
    }


def decode_rle(rle: Dict[str, Any]) -> np.ndarray:
    """
    Decode RLE (Run-Length Encoding) to a binary mask.

    Args:
        rle: Dict with 'counts' (RLE string) and 'size' [height, width]

    Returns:
        2D binary numpy array (H, W)
    """
    from pycocotools import mask as mask_utils

    # Prepare RLE dict for pycocotools
    rle_dict = {
        "counts": rle["counts"].encode("utf-8")
        if isinstance(rle["counts"], str)
        else rle["counts"],
        "size": rle["size"],
    }

    # Decode
    mask = mask_utils.decode(rle_dict)

    return mask.astype(bool)


def mask_to_bbox(mask: np.ndarray) -> List[int]:
    """
    Compute bounding box from a binary mask.

    Args:
        mask: 2D binary numpy array (H, W)

    Returns:
        Bounding box [x1, y1, x2, y2] in XYXY format (pixels)
        Returns [0, 0, 0, 0] if mask is empty
    """
    # Find non-zero indices
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        return [0, 0, 0, 0]

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    # Return XYXY format
    return [int(x_min), int(y_min), int(x_max), int(y_max)]


def masks_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU) between two masks.

    Args:
        mask1: First binary mask (H, W)
        mask2: Second binary mask (H, W)

    Returns:
        IoU score between 0 and 1
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 0.0

    return float(intersection / union)


def bbox_iou(box1: List[int], box2: List[int]) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: First box [x1, y1, x2, y2] in XYXY format
        box2: Second box [x1, y1, x2, y2] in XYXY format

    Returns:
        IoU score between 0 and 1
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 < x1 or y2 < y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return float(intersection / union)


def mask_area(mask: np.ndarray) -> int:
    """
    Compute the area (number of pixels) of a binary mask.

    Args:
        mask: 2D binary numpy array (H, W)

    Returns:
        Number of True/non-zero pixels
    """
    return int(np.sum(mask))


def crop_mask(mask: np.ndarray, bbox: List[int]) -> np.ndarray:
    """
    Crop a mask to a bounding box region.

    Args:
        mask: 2D binary numpy array (H, W)
        bbox: Bounding box [x1, y1, x2, y2] in XYXY format

    Returns:
        Cropped mask
    """
    x1, y1, x2, y2 = bbox
    return mask[y1:y2, x1:x2].copy()


def resize_mask(mask: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize a binary mask to target size.

    Args:
        mask: 2D binary numpy array (H, W)
        target_size: Target size (height, width)

    Returns:
        Resized mask
    """
    from PIL import Image

    # Convert to PIL, resize, convert back
    mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
    mask_img = mask_img.resize((target_size[1], target_size[0]), Image.NEAREST)

    return np.array(mask_img) > 127


def combine_masks(masks: List[np.ndarray]) -> np.ndarray:
    """
    Combine multiple binary masks into one using logical OR.

    Args:
        masks: List of 2D binary numpy arrays (same shape)

    Returns:
        Combined mask
    """
    if not masks:
        raise ValueError("No masks provided")

    combined = masks[0].copy()
    for mask in masks[1:]:
        combined = np.logical_or(combined, mask)

    return combined


def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    """
    Convert binary mask to polygon coordinates.

    Args:
        mask: 2D binary numpy array (H, W)

    Returns:
        List of polygons, each polygon is [x1, y1, x2, y2, ...] flattened coords
    """
    import cv2

    mask_uint8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    polygons = []
    for contour in contours:
        if len(contour) >= 3:  # Valid polygon needs at least 3 points
            polygon = contour.flatten().tolist()
            polygons.append(polygon)

    return polygons
