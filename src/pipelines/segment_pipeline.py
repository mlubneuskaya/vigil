from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment

from tasks.entity_extractor import EntityExtractor
from engines.qwen import QwenEngine
from engines.sam import SAM3Segmenter
from engines.dino_embedder import DINOv3Embedder
from utils.mask_utils import encode_rle, decode_rle
from utils.visualization import draw_results, save_visualization
from configs.config_schema import SegmentationConfig, EntityExtractorConfig

logger = logging.getLogger(__name__)


class SegmentationPipeline:
    """
    Complete segmentation pipeline using SAM3 and Qwen3-VL.
    Optimized for throughput and VRAM efficiency using a streaming buffer.
    """

    def __init__(
        self,
        qwen_engine: QwenEngine,
        config: SegmentationConfig,
        entity_extractor_config: EntityExtractorConfig,
    ):
        """
        Initialize the segmentation pipeline.
        """
        self.config = config
        self.device = config.device

        logger.info("Initializing Segmentation Pipeline...")
        logger.info("=" * 50)

        self.text_parser = EntityExtractor(
            engine=qwen_engine, config=entity_extractor_config
        )

        self.segmenter = SAM3Segmenter(
            model_name=config.segmentation_model,
            dtype=config.dtype,
            device=config.device,
        )

        self.dino_embedder = DINOv3Embedder(
            model_name=config.dino_model, device=config.device
        )

        logger.info("=" * 50)
        logger.info(f"Pipeline ready! (Batch Size: {config.dino_batch_size})")

    def run_batch(
        self,
        datapoints: List[Dict],
        pair_objects: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Run the pipeline on a list of datapoints.
        """
        logger.info(f"Processing {len(datapoints)} datapoints...")

        logger.info(f"[1/3] Extracting entities...")
        text_prompts = [dp["prompt"] for dp in datapoints]
        all_entities = self.text_parser.extract_entities(text_prompts)

        crop_buffer: List[Image.Image] = []
        object_ref_buffer: List[Dict] = []

        dp_results = []

        logger.info(f"[2/3] Segmenting and embedding (Streaming Mode)...")

        for i, (dp, entities) in enumerate(zip(datapoints, all_entities)):
            dp_result = {
                "id": dp["id"],
                "contextual_results": [],
                "generated_result": None,
            }

            for img_idx, img_path in enumerate(dp["object_images"]):
                res = self._process_single_image(
                    img_path, entities, crop_buffer, object_ref_buffer
                )
                dp_result["contextual_results"].append(res)

                if len(crop_buffer) >= self.config.dino_batch_size:
                    self._flush_dino_buffer(crop_buffer, object_ref_buffer)

                if self.config.save_viz:
                    self._save_viz(res, dp["id"], f"contextual_{img_idx}")
                    self._save_crops(
                        res["path"], res["objects"], dp["id"], f"contextual_{img_idx}"
                    )

            gen_res = self._process_single_image(
                dp["generated_image"], entities, crop_buffer, object_ref_buffer
            )
            dp_result["generated_result"] = gen_res

            if len(crop_buffer) >= self.config.dino_batch_size:
                self._flush_dino_buffer(crop_buffer, object_ref_buffer)

            if self.config.save_viz:
                self._save_viz(gen_res, dp["id"], f"generated")
                self._save_crops(
                    gen_res["path"], gen_res["objects"], dp["id"], f"generated"
                )

            dp_results.append(dp_result)

        if crop_buffer:
            self._flush_dino_buffer(crop_buffer, object_ref_buffer)

        logger.info("[3/3] Pairing objects...")

        final_output = []
        for i, dp_data in enumerate(dp_results):
            pairings = []

            if pair_objects:
                pairings = self.pair_objects_optimized(
                    dp_data["contextual_results"], dp_data["generated_result"]
                )

            self._remove_tensors(dp_data["contextual_results"])
            self._remove_tensors([dp_data["generated_result"]])

            final_output.append(
                {
                    "id": dp_data["id"],
                    "text_prompt": datapoints[i]["prompt"],
                    "detected_entities": all_entities[i],
                    "contextual_results": dp_data["contextual_results"],
                    "generated_result": dp_data["generated_result"],
                    "object_pairings": pairings,
                }
            )

        return final_output

    def _flush_dino_buffer(self, crops: List[Image.Image], refs: List[Dict]):
        """
        Runs DINO on the current buffer and clears it.
        This keeps RAM usage constant regardless of dataset size.
        """
        if not crops:
            return

        embeddings = self.dino_embedder.get_embedding(crops)

        for i, emb in enumerate(embeddings):
            refs[i]["embedding_tensor"] = emb

        crops.clear()
        refs.clear()

    def _process_single_image(
        self,
        image_path: str,
        labels: List[str],
        crop_accumulator: List[Image.Image],
        ref_accumulator: List[Dict],
    ) -> Dict[str, Any]:
        """
        Loads, segments, and crops a single image.
        Appends crops and object references to global accumulators for batching.
        """
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return {"path": image_path, "objects": [], "error": str(e)}

        filename_no_ext = Path(image_path).stem

        width, height = image.size

        raw_results = self.segmenter.segment(
            image=image,
            labels=labels,
            threshold=self.config.segmentation_threshold,
            mask_threshold=self.config.mask_threshold,
            iou_threshold=self.config.iou_threshold,
        )

        formatted_objects = []

        for obj in raw_results:
            mask = obj["mask"]

            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            if not np.any(rows) or not np.any(cols):
                continue

            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]

            crop = image.crop((x_min, y_min, x_max + 1, y_max + 1))

            crop_mask = mask[y_min : y_max + 1, x_min : x_max + 1]

            crop_arr = np.array(crop)
            crop_arr[~crop_mask] = 0
            masked_crop = Image.fromarray(crop_arr.astype(np.uint8))

            base_obj_id = obj.get("object_id", len(formatted_objects))
            object_id_with_filename = f"{filename_no_ext}_{base_obj_id}"

            obj_dict = {
                "object_id": object_id_with_filename,
                "label": obj["label"],
                "score": float(obj["score"]),
                "mask": encode_rle(mask),
                "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)],
                "embedding_tensor": None,
            }

            formatted_objects.append(obj_dict)

            crop_accumulator.append(masked_crop)
            ref_accumulator.append(obj_dict)

        return {
            "path": str(Path(image_path).absolute()),
            "image_dimensions": {"width": width, "height": height},
            "objects": formatted_objects,
        }

    def pair_objects_optimized(
        self, contextual_results, generated_result
    ) -> List[Dict]:
        """
        Pair objects using optimal bipartite matching (Hungarian algorithm).
        Ensures that when multiple similar objects exist, assignments are globally optimal
        rather than greedy.
        """
        ctx_objs = [
            {**obj, "source_image": ctx["path"]}
            for ctx in contextual_results
            for obj in ctx["objects"]
            if obj.get("embedding_tensor") is not None
        ]

        gen_objs = [
            obj
            for obj in generated_result["objects"]
            if obj.get("embedding_tensor") is not None
        ]

        pairs = []

        if not ctx_objs:
            return []

        if not gen_objs:
            return []

        ctx_embs = torch.stack([o["embedding_tensor"] for o in ctx_objs])
        gen_embs = torch.stack([o["embedding_tensor"] for o in gen_objs])

        ctx_embs = F.normalize(ctx_embs, p=2, dim=1)
        gen_embs = F.normalize(gen_embs, p=2, dim=1)

        sim_matrix = torch.mm(ctx_embs, gen_embs.T)

        cost_matrix = np.full((len(ctx_objs), len(gen_objs)), 1000.0)

        for i, ctx_obj in enumerate(ctx_objs):
            for j, gen_obj in enumerate(gen_objs):
                if ctx_obj["label"] == gen_obj["label"]:
                    cost_matrix[i, j] = -float(sim_matrix[i, j].item())

        ctx_indices, gen_indices = linear_sum_assignment(cost_matrix)

        assignment_map = dict(zip(ctx_indices, gen_indices))

        for i, ctx_obj in enumerate(ctx_objs):
            if i in assignment_map:
                j = assignment_map[i]
                best_sim = float(sim_matrix[i, j].item())
                best_gen = gen_objs[j]
            else:
                best_sim = -1.0
                best_gen = None

            pair_entry = self._create_pair_entry(ctx_obj, best_gen, best_sim)

            if pair_entry["paired"]:
                pairs.append(pair_entry)

        return pairs

    def _create_pair_entry(self, ctx_obj, gen_obj, score) -> Dict:
        return {
            "contextual_object_id": ctx_obj["object_id"],
            "contextual_label": ctx_obj["label"],
            "source_image": ctx_obj["source_image"],
            "best_generated_object_id": gen_obj["object_id"] if gen_obj else None,
            "best_generated_label": gen_obj["label"] if gen_obj else None,
            "cosine_similarity": score,
            "threshold": self.config.pairing_threshold,
            "paired": score >= self.config.pairing_threshold if gen_obj else False,
        }

    def _remove_tensors(self, result_list: List[Dict]):
        """
        Removes Tensor objects to ensure the dictionary is JSON serializable.
        """
        for res in result_list:
            if not res or "objects" not in res:
                continue
            for obj in res["objects"]:
                if "embedding_tensor" in obj:
                    del obj["embedding_tensor"]

    def _save_viz(self, result, dp_id, prefix):
        """Helper to save visualization images."""
        if not result.get("objects"):
            return

        viz_objs = []
        for obj in result["objects"]:
            o_copy = obj.copy()
            o_copy["mask"] = decode_rle(obj["mask"])
            viz_objs.append(o_copy)

        try:
            image = Image.open(result["path"]).convert("RGB")
            annotated = draw_results(image, viz_objs)
            output_dir = f"{self.config.output_dir}/{dp_id}/{self.config.viz_dir}"
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            save_visualization(annotated, f"{output_dir}/{prefix}_annotated.jpg")
        except Exception as e:
            logger.warning(f"Could not save viz for {result['path']}: {e}")

    def _save_crops(
        self, image_path: str, objects: List[Dict], dp_id: str, prefix: str
    ):
        """Helper to save individual object crops as images."""
        if not objects:
            return

        try:
            image = Image.open(image_path).convert("RGB")
            output_base_dir = (
                f"{self.config.output_dir}/{dp_id}/{self.config.viz_dir}/{prefix}_crops"
            )
            Path(output_base_dir).mkdir(parents=True, exist_ok=True)

            for obj_idx, obj in enumerate(objects):
                label = obj.get("label", "object").replace(" ", "_").lower()
                bbox = obj["bbox"]
                x_min, y_min, x_max, y_max = bbox

                crop = image.crop((x_min, y_min, x_max + 1, y_max + 1))
                crop_path = f"{output_base_dir}/{obj_idx:03d}_{label}_crop.jpg"
                save_visualization(crop, crop_path)

        except Exception as e:
            logger.warning(f"Could not save crops for {image_path}: {e}")

    @staticmethod
    def to_json(
        output: Dict[str, Any],
        output_path: Optional[str] = None,
        indent: int = 4,
    ) -> str:
        """Convert pipeline output to JSON."""
        json_str = json.dumps(output, indent=indent, ensure_ascii=False)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(json_str)

        return json_str
