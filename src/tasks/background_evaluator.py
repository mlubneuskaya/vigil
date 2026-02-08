import torch
import numpy as np
import json
import logging
import cv2
from pathlib import Path
from PIL import Image
from typing import List, Dict, Any, Tuple, Optional, Union

from engines.qwen import QwenEngine
from utils.mask_utils import decode_rle
from configs.config_schema import BackgroundEvaluatorConfig

logger = logging.getLogger(__name__)


class BackgroundDescriptionEvaluator:
    """
    Handles qualitative analysis using the shared VLM engine.
    """

    def __init__(self, engine: QwenEngine, config: BackgroundEvaluatorConfig):
        self.engine = engine
        self.config = config

    def analyze_batch(self, pairs: List[Tuple[Image.Image, Image.Image]]) -> List[str]:
        """
        Generates descriptions for differences in a batch of image pairs.
        """
        if not pairs:
            return []

        prompts = [self.config.system_prompt] * len(pairs)
        max_dim = self.config.vlm_resolution

        def fast_resize(img):
            w, h = img.size
            if max(w, h) > max_dim:
                scale = max_dim / max(w, h)
                new_w, new_h = int(w * scale), int(h * scale)
                return img.resize((new_w, new_h), Image.BILINEAR)
            return img

        batch_images = [[fast_resize(ref), fast_resize(gen)] for ref, gen in pairs]

        responses = self.engine.generate(prompts, images=batch_images)
        return responses


def highlight_differences(
    img_a: Image.Image, img_b: Image.Image, config
) -> Tuple[Image.Image, Image.Image, bool, Optional[Image.Image]]:
    """
    Detects differences using SSIM and draws bounding boxes around them.
    Uses configuration from DifferenceHighlightConfig.
    """
    if not config.enabled:
        return img_a, img_b, False, None

    if img_a.size != img_b.size:
        img_b = img_b.resize(img_a.size, Image.BILINEAR)

    a_cv = cv2.cvtColor(np.array(img_a), cv2.COLOR_RGB2BGR)
    b_cv = cv2.cvtColor(np.array(img_b), cv2.COLOR_RGB2BGR)

    gray_a = cv2.cvtColor(a_cv, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(b_cv, cv2.COLOR_BGR2GRAY)

    gray_a = cv2.GaussianBlur(
        gray_a, (config.gaussian_blur_kernel, config.gaussian_blur_kernel), 0
    )
    gray_b = cv2.GaussianBlur(
        gray_b, (config.gaussian_blur_kernel, config.gaussian_blur_kernel), 0
    )

    diff = cv2.absdiff(gray_a, gray_b)

    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (config.morph_kernel_size, config.morph_kernel_size)
    )
    thresh = cv2.dilate(thresh, kernel, iterations=config.dilation_iterations)

    cnts, _ = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    change_detected = False
    diff_pil = Image.fromarray(thresh)

    h_img, w_img = thresh.shape
    total_pixels = w_img * h_img
    min_area = int(total_pixels * config.min_area_ratio)

    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        change_detected = True
        (x, y, w, h) = cv2.boundingRect(c)

        pad = config.padding
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(a_cv.shape[1] - x, w + 2 * pad)
        h = min(a_cv.shape[0] - y, h + 2 * pad)

        cv2.rectangle(
            a_cv, (x, y), (x + w, y + h), config.box_color_bgr, config.box_thickness
        )
        cv2.rectangle(
            b_cv, (x, y), (x + w, y + h), config.box_color_bgr, config.box_thickness
        )

    out_a = Image.fromarray(cv2.cvtColor(a_cv, cv2.COLOR_BGR2RGB))
    out_b = Image.fromarray(cv2.cvtColor(b_cv, cv2.COLOR_BGR2RGB))

    return out_a, out_b, change_detected, diff_pil


def process_background_evaluation(
    input_data: Union[Dict[str, Any], List[Dict[str, Any]]],
    qwen_evaluator: BackgroundDescriptionEvaluator,
    output_path: Optional[str] = None,
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Pipeline Step 3: Two-Stage Background Consistency Check.
    Memory Safe Version: Processes and flushes images in batches.
    """

    bg_eval_config = qwen_evaluator.config
    batch_size = bg_eval_config.batch_size

    total = len(input_data)

    for i in range(0, total, batch_size):
        batch_indices = range(i, min(i + batch_size, total))
        batch_jobs = []

        for idx in batch_indices:
            entry = input_data[idx]

            gen_data = entry.get("generated_result")
            gen_path = gen_data.get("path")

            paired_ids = {
                pair["best_generated_object_id"]
                for pair in entry.get("object_pairings", [])
                if pair.get("paired") is True
            }

            objects = [
                obj for obj in gen_data["objects"] if obj["object_id"] in paired_ids
            ]

            parent_dir = Path(gen_path).parent
            bg_candidates = list(parent_dir.glob("*background*.png"))

            if not bg_candidates and entry.get("background_image"):
                bg_candidates = [Path(entry.get("background_image"))]

            if bg_candidates:
                bg_path = str(bg_candidates[0])
                try:
                    ref_img_clean, gen_img_clean = prepare_masked_images(
                        gen_path, objects, bg_path, bg_eval_config.mask_margin_percent
                    )

                    ref_img_boxed, gen_img_boxed, has_change, ssim_diff = (
                        highlight_differences(
                            ref_img_clean,
                            gen_img_clean,
                            bg_eval_config.absolute_difference_config,
                        )
                    )

                    batch_jobs.append(
                        {
                            "entry_idx": idx,
                            "ref_img": ref_img_boxed,
                            "gen_img": gen_img_boxed,
                            "bg_path": bg_path,
                            "has_change": has_change,
                            "ssim_diff": ssim_diff,
                        }
                    )

                    if bg_eval_config.save_viz:
                        directory = f"{bg_eval_config.output_dir}/{entry['id']}/{bg_eval_config.viz_dir}"
                        save_visualization(
                            ref_img_boxed, directory, f"reference_masked_boxed.jpg"
                        )
                        save_visualization(
                            gen_img_boxed, directory, f"generated_masked_boxed.jpg"
                        )
                        if ssim_diff:
                            save_visualization(
                                ssim_diff, directory, f"ssim_difference_map.jpg"
                            )

                except Exception as e:
                    logger.error(f"Error prep {gen_path}: {e}")
            else:
                entry["background_evaluation"] = {
                    "status": "skipped",
                    "reason": "no_background_file",
                }

        if not batch_jobs:
            continue

        batch_jobs.sort(key=lambda x: x["ref_img"].size[0] * x["ref_img"].size[1])

        image_pairs = [(job["ref_img"], job["gen_img"]) for job in batch_jobs]

        analyses = qwen_evaluator.analyze_batch(image_pairs)

        for k, analysis in enumerate(analyses):
            batch_jobs[k]["qwen_analysis"] = analysis

        for job in batch_jobs:
            idx = job["entry_idx"]

            input_data[idx]["background_evaluation"] = {
                "status": "success",
                "reference_path": job["bg_path"],
                "qwen_analysis": job.get("qwen_analysis", None),
                "change_detected_by_ssim": job.get("has_change", False),
            }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(input_data, f, indent=4, default=str)

    return input_data


def prepare_masked_images(
    gen_path: str, objects: List[Dict], ref_path: str, margin_percent: float = 0.0
) -> Tuple[Image.Image, Image.Image]:
    """Loads images and blacks out the foreground objects with optional margin."""
    gen_img = Image.open(gen_path).convert("RGB")
    ref_img = Image.open(ref_path).convert("RGB")

    combined_mask = _create_combined_mask(objects, gen_img.width, gen_img.height, margin_percent)

    masked_gen = _apply_mask(gen_img, combined_mask)
    masked_ref = _apply_mask(ref_img, combined_mask)

    return masked_ref, masked_gen


def _create_combined_mask(objects, width, height, margin_percent: float = 0.0):
    """Creates a combined mask using bboxes from SAM with optional margin.
    
    Args:
        objects: List of objects with bbox coordinates
        width: Image width
        height: Image height
        margin_percent: Margin as percentage of object size (0.0-1.0)
    """
    combined = np.zeros((height, width), dtype=bool)
    
    for obj in objects:
        if "bbox" in obj:
            x_min, y_min, x_max, y_max = obj["bbox"]
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min
            
            # Calculate margin in pixels
            margin_x = int(bbox_width * margin_percent / 2)
            margin_y = int(bbox_height * margin_percent / 2)
            
            # Apply margin with bounds checking
            x_min_expanded = max(0, x_min - margin_x)
            y_min_expanded = max(0, y_min - margin_y)
            x_max_expanded = min(width, x_max + margin_x)
            y_max_expanded = min(height, y_max + margin_y)
            
            combined[y_min_expanded:y_max_expanded + 1, x_min_expanded:x_max_expanded + 1] = True
    
    return combined


def _apply_mask(image, mask):
    img_arr = np.array(image)
    if mask.shape[:2] != img_arr.shape[:2]:
        mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
        mask_img = mask_img.resize((img_arr.shape[1], img_arr.shape[0]), Image.NEAREST)
        mask = np.array(mask_img) > 0

    final_img = img_arr.copy()
    final_img[mask] = 0
    return Image.fromarray(final_img)


def save_visualization(image: Image.Image, viz_dir: str, filename: str):
    """
    Helper to save visualization images.
    Ensures directory exists and handles file extensions.
    """
    try:
        output_dir = Path(viz_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        save_path = output_dir / filename
        image.save(save_path)
        logger.debug(f"Saved visualization to {save_path}")

    except Exception as e:
        logger.warning(f"Could not save visualization for {filename}: {e}")
