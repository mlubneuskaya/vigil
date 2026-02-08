"""
Visualization Module

Provides functions to draw segmentation results (masks and bounding boxes)
on images using the supervision library.
"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import numpy as np
from PIL import Image

# Color palette for different objects
DEFAULT_COLORS = [
    (255, 0, 0),  # Red
    (0, 255, 0),  # Green
    (0, 0, 255),  # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (255, 128, 0),  # Orange
    (128, 0, 255),  # Purple
    (0, 255, 128),  # Spring green
    (255, 0, 128),  # Rose
]


def draw_results(
    image: Image.Image,
    objects: List[Dict[str, Any]],
    draw_masks: bool = True,
    draw_boxes: bool = True,
    draw_labels: bool = True,
    mask_opacity: float = 0.4,
    box_thickness: int = 2,
    colors: Optional[List[Tuple[int, int, int]]] = None,
) -> Image.Image:
    """
    Draw segmentation results on an image.

    Args:
        image: PIL Image to annotate
        objects: List of detected objects with 'label', 'bbox', 'mask', 'score'
        draw_masks: Whether to draw segmentation masks
        draw_boxes: Whether to draw bounding boxes
        draw_labels: Whether to draw labels with scores
        mask_opacity: Opacity for mask overlay (0-1)
        box_thickness: Thickness of bounding box lines
        colors: Optional custom color palette

    Returns:
        Annotated PIL Image
    """
    import supervision as sv

    if colors is None:
        colors = DEFAULT_COLORS

    # Convert PIL to numpy for supervision
    image_np = np.array(image)

    if not objects:
        return image

    # Prepare supervision Detections
    xyxy = np.array([obj["bbox"] for obj in objects])
    masks = np.array([obj["mask"] for obj in objects]) if draw_masks else None
    confidence = np.array([obj.get("score", 1.0) for obj in objects])
    class_ids = np.arange(len(objects))

    # Create labels
    labels = [f"{obj['object_id']}: {obj.get('score', 1.0):.2f}" for obj in objects]

    detections = sv.Detections(
        xyxy=xyxy,
        mask=masks,
        confidence=confidence,
        class_id=class_ids,
    )

    # Create annotators
    annotated = image_np.copy()

    if draw_masks and masks is not None:
        mask_annotator = sv.MaskAnnotator(
            opacity=mask_opacity,
        )
        annotated = mask_annotator.annotate(
            scene=annotated,
            detections=detections,
        )

    if draw_boxes:
        box_annotator = sv.BoxAnnotator(
            thickness=box_thickness,
        )
        annotated = box_annotator.annotate(
            scene=annotated,
            detections=detections,
        )

    if draw_labels:
        label_annotator = sv.LabelAnnotator(
            text_position=sv.Position.TOP_LEFT,
        )
        annotated = label_annotator.annotate(
            scene=annotated,
            detections=detections,
            labels=labels,
        )

    return Image.fromarray(annotated)


def draw_comparison(
    contextual_image: Image.Image,
    contextual_objects: List[Dict[str, Any]],
    generated_image: Image.Image,
    generated_objects: List[Dict[str, Any]],
    title: Optional[str] = None,
) -> Image.Image:
    """
    Create a side-by-side comparison of contextual and generated images.

    Args:
        contextual_image: Source contextual image
        contextual_objects: Objects detected in contextual image
        generated_image: Generated image
        generated_objects: Objects detected in generated image
        title: Optional title for the comparison

    Returns:
        Combined comparison image
    """
    from PIL import ImageDraw, ImageFont

    # Draw annotations on both images
    ctx_annotated = draw_results(contextual_image, contextual_objects)
    gen_annotated = draw_results(generated_image, generated_objects)

    # Resize to same height
    max_height = max(ctx_annotated.height, gen_annotated.height)

    if ctx_annotated.height != max_height:
        ratio = max_height / ctx_annotated.height
        new_width = int(ctx_annotated.width * ratio)
        ctx_annotated = ctx_annotated.resize((new_width, max_height))

    if gen_annotated.height != max_height:
        ratio = max_height / gen_annotated.height
        new_width = int(gen_annotated.width * ratio)
        gen_annotated = gen_annotated.resize((new_width, max_height))

    # Create combined image
    padding = 20
    title_height = 40 if title else 0
    combined_width = ctx_annotated.width + gen_annotated.width + padding
    combined_height = max_height + title_height

    combined = Image.new("RGB", (combined_width, combined_height), (255, 255, 255))

    # Paste images
    y_offset = title_height
    combined.paste(ctx_annotated, (0, y_offset))
    combined.paste(gen_annotated, (ctx_annotated.width + padding, y_offset))

    # Add title if provided
    if title:
        draw = ImageDraw.Draw(combined)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        draw.text((10, 10), title, fill=(0, 0, 0), font=font)

    return combined


def save_visualization(
    image: Image.Image,
    output_path: str,
    quality: int = 95,
) -> str:
    """
    Save visualization to file.

    Args:
        image: PIL Image to save
        output_path: Path to save the image
        quality: JPEG quality (1-100)

    Returns:
        Absolute path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine format from extension
    suffix = output_path.suffix.lower()
    if suffix in [".jpg", ".jpeg"]:
        image.save(output_path, "JPEG", quality=quality)
    elif suffix == ".png":
        image.save(output_path, "PNG")
    else:
        image.save(output_path)

    return str(output_path.absolute())


def save_mask_image(
    mask: np.ndarray,
    output_path: str,
) -> str:
    """
    Save a binary mask as an image.

    Args:
        mask: 2D binary numpy array
        output_path: Path to save the mask image

    Returns:
        Absolute path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert bool mask to uint8
    mask_uint8 = mask.astype(np.uint8) * 255
    mask_image = Image.fromarray(mask_uint8, mode="L")
    mask_image.save(output_path)

    return str(output_path.absolute())


def create_masked_image(
    image: Image.Image,
    mask: np.ndarray,
    background_color: Tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """
    Create an image with only the masked region visible.

    Useful for creating inputs for Qwen3-VL analysis of specific objects.

    Args:
        image: Original PIL Image
        mask: Binary mask array
        background_color: Color for non-masked regions

    Returns:
        Image with only masked region visible
    """
    image_np = np.array(image)

    # Create background
    background = np.full_like(image_np, background_color)

    # Apply mask
    mask_3d = np.stack([mask] * 3, axis=-1)
    result = np.where(mask_3d, image_np, background)

    return Image.fromarray(result.astype(np.uint8))


def create_highlighted_image(
    image: Image.Image,
    mask: np.ndarray,
    highlight_color: Tuple[int, int, int] = (255, 0, 0),
    highlight_opacity: float = 0.3,
    dim_background: bool = True,
    dim_factor: float = 0.5,
) -> Image.Image:
    """
    Create an image with the masked region highlighted.

    Args:
        image: Original PIL Image
        mask: Binary mask array
        highlight_color: Color for highlight overlay
        highlight_opacity: Opacity of highlight (0-1)
        dim_background: Whether to dim non-masked regions
        dim_factor: How much to dim background (0-1)

    Returns:
        Image with highlighted region
    """
    image_np = np.array(image).astype(float)

    # Dim background if requested
    if dim_background:
        mask_3d = np.stack([mask] * 3, axis=-1)
        image_np = np.where(mask_3d, image_np, image_np * dim_factor)

    # Add highlight overlay
    highlight = np.array(highlight_color)
    mask_3d = np.stack([mask] * 3, axis=-1)
    overlay = mask_3d * highlight * highlight_opacity
    image_np = np.clip(image_np + overlay, 0, 255)

    return Image.fromarray(image_np.astype(np.uint8))
