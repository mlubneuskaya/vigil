import json
import re
import logging
from pathlib import Path
from PIL import Image
from typing import Dict, Any, List, Optional, Union
import numpy as np


from engines.qwen import QwenEngine
from configs.config_schema import ObjectEvaluatorConfig
from utils.visualization import save_visualization

logger = logging.getLogger(__name__)


class PairedObjectEvaluator:
    """
    Compares recontextualized objects with their original versions using Qwen3-VL
    to detect hallucinations and assess visual consistency.
    """

    def __init__(self, engine: QwenEngine, config: ObjectEvaluatorConfig):
        self.engine = engine
        self.config = config
        logger.info(
            f"PairedObjectEvaluator initialized with config: {config.model_dump()}"
        )

    def evaluate_batch(self, batch_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not batch_pairs:
            return []

        prompts = []
        batch_images = []

        for idx, item in enumerate(batch_pairs):
            prompt = self._build_prompt()
            prompts.append(prompt)

            ctx_img = item["ctx_img"]
            gen_img = item["gen_img"]

            batch_images.append([ctx_img, gen_img])

        try:
            responses = self.engine.generate(prompts, images=batch_images)
        except Exception as e:
            logger.error(f"Batch generation failed: {e}")
            return [
                {"overall_assessment": "ERROR", "error": str(e)} for _ in batch_pairs
            ]

        results = []
        for idx, response in enumerate(responses):
            parsed = self._parse_response(response)
            results.append(parsed)

        return results

    def extract_crop(self, image_path: str, bbox: List[int]) -> Optional[Image.Image]:
        """Helper to load an image and crop to the bbox with padding."""
        try:
            image_path = str(image_path)
            if not Path(image_path).exists():
                logger.warning(f"Image not found: {image_path}")
                return None

            image = Image.open(image_path).convert("RGB")

            if not bbox or len(bbox) != 4:
                return self._resize(image)

            x1, y1, x2, y2 = bbox
            w, h = image.size

            pad = self.config.padding

            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)

            crop = image.crop((x1, y1, x2, y2))
            return self._resize(crop)
        except Exception as e:
            logger.error(f"Error cropping {image_path}: {e}")
            return None

    def _build_prompt(self) -> str:
        prompt_template = self.config.prompt_template
        return prompt_template

    def _parse_response(self, text: str) -> Dict[str, Any]:
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            pass

        return {
            "overall_assessment": "ERROR",
            "overall_score": 0,
            "raw_response": text,
            "error": "Failed to parse JSON",
        }

    def _resize(self, img: Image.Image) -> Image.Image:
        """
        Downscales image if it exceeds the configured resolution.
        This ensures batch inference stays fast.
        """
        max_dim = self.config.resolution

        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            return img.resize((new_w, new_h), Image.BILINEAR)

        return img

    def _save_debug_images(
        self, ctx_img: Image.Image, gen_img: Image.Image, dp_idx: int, pairing_idx: int
    ):
        """Save contextual and generated images for debugging."""
        if not self.config.save_viz:
            return

        try:
            output_dir = f"{self.config.output_dir}/debug_pairs"
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            ctx_path = (
                f"{output_dir}/dp_{dp_idx:04d}_pair_{pairing_idx:02d}_contextual.jpg"
            )
            gen_path = (
                f"{output_dir}/dp_{dp_idx:04d}_pair_{pairing_idx:02d}_generated.jpg"
            )

            save_visualization(ctx_img, ctx_path)
            save_visualization(gen_img, gen_path)
        except Exception as e:
            logger.warning(
                f"Could not save debug images for dp {dp_idx} pair {pairing_idx}: {e}"
            )


def process_object_consistency(
    input_data: Union[Dict[str, Any], List[Dict[str, Any]]],
    evaluator: PairedObjectEvaluator,
    output_path: Optional[str] = None,
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Pipeline Step 4: Object Hallucination & Consistency Check.
    """

    batch_size = evaluator.config.batch_size

    job_buffer = []

    for dp_idx, dp in enumerate(input_data):
        obj_map = {}
        for ctx_res in dp.get("contextual_results", []):
            path = ctx_res.get("path")
            for obj in ctx_res.get("objects", []):
                if "object_id" in obj:
                    obj_map[f"ctx_{obj['object_id']}"] = {
                        "path": path,
                        "bbox": obj.get("bbox"),
                        "label": obj.get("label", "unknown"),
                    }

        gen_res = dp.get("generated_result")
        if gen_res:
            path = gen_res.get("path")
            for obj in gen_res.get("objects", []):
                if "object_id" in obj:
                    obj_map[f"gen_{obj['object_id']}"] = {
                        "path": path,
                        "bbox": obj.get("bbox"),
                        "label": obj.get("label", "unknown"),
                    }

        pairings = dp.get("object_pairings", [])
        if not pairings:
            continue

        for p_idx, pair in enumerate(pairings):
            ctx_id = pair.get("contextual_object_id")
            gen_id = pair.get("best_generated_object_id") or pair.get(
                "generated_object_id"
            )

            if not ctx_id or not gen_id:
                continue

            ctx_info = obj_map.get(f"ctx_{ctx_id}")
            gen_info = obj_map.get(f"gen_{gen_id}")

            if ctx_info and gen_info:
                ctx_crop = evaluator.extract_crop(ctx_info["path"], ctx_info["bbox"])
                gen_crop = evaluator.extract_crop(gen_info["path"], gen_info["bbox"])

                if ctx_crop and gen_crop:
                    job_buffer.append(
                        {
                            "dp_idx": dp_idx,
                            "pairing_idx": p_idx,
                            "ctx_img": ctx_crop,
                            "gen_img": gen_crop,
                            "ctx_label": ctx_info["label"],
                            "gen_label": gen_info["label"],
                        }
                    )

            if len(job_buffer) >= batch_size:
                _process_buffer(job_buffer, evaluator, input_data)
                job_buffer = []

    if job_buffer:
        _process_buffer(job_buffer, evaluator, input_data)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(input_data, f, indent=4, default=str)

    return input_data


def _process_buffer(buffer, evaluator, datapoints):
    if not buffer:
        return

    for job in buffer:
        evaluator._save_debug_images(
            job["ctx_img"], job["gen_img"], job["dp_idx"], job["pairing_idx"]
        )

    results = evaluator.evaluate_batch(buffer)
    for job, result in zip(buffer, results):
        dp = datapoints[job["dp_idx"]]
        dp["object_pairings"][job["pairing_idx"]]["hallucination_check"] = result
