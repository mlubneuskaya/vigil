import json
import re
from typing import List
import logging

from engines.qwen import QwenEngine
from configs.config_schema import EntityExtractorConfig

logger = logging.getLogger(__name__)


class EntityExtractor:
    """
    Extracts objects/entities from text prompts using Qwen3-VL-8B-Instruct.
    """

    def __init__(self, engine: QwenEngine, config: EntityExtractorConfig):
        """
        Args:
            engine: The shared QwenEngine instance.
            config: Configuration containing prompts and exclusion lists.
        """
        self.engine = engine
        self.config = config

        self.excluded_terms = {t.lower().strip() for t in config.excluded_terms}
        self.color_modifiers = {c.lower().strip() for c in config.color_modifiers}

    def extract_entities(self, prompts: List[str]) -> List[List[str]]:
        """
        Generates entity lists for a batch of prompts.
        """
        all_entities = []

        batch_size = self.config.batch_size

        total = len(prompts)

        for i in range(0, total, batch_size):
            batch_prompts_raw = prompts[i : i + batch_size]

            formatted_batch = [
                self.config.extraction_prompt.format(prompt=p)
                for p in batch_prompts_raw
            ]

            try:
                batch_responses = self.engine.generate(formatted_batch)
            except Exception as e:
                logger.error(f"Batch generation failed at index {i}: {e}")
                batch_responses = [""] * len(batch_prompts_raw)

            for resp in batch_responses:
                entities = self._parse_entities(resp)
                filtered = self._filter_excluded_entities(entities)
                all_entities.append(filtered)

        return all_entities

    def _filter_excluded_entities(self, entities: List[str]) -> List[str]:
        """
        Filter out non-physical/abstract entities based on config rules.
        """
        filtered = []
        for entity in entities:
            entity_lower = entity.lower().strip()

            if entity_lower in self.excluded_terms:
                continue

            is_excluded = False
            for term in self.excluded_terms:
                if entity_lower == term or entity_lower.endswith(" " + term):
                    is_excluded = True
                    break

                if term in entity_lower:
                    words = entity_lower.split()
                    if words[-1] == term and words[0] in self.color_modifiers:
                        is_excluded = True
                        break

            if not is_excluded:
                filtered.append(entity)

        return filtered

    def _parse_entities(self, response: str) -> List[str]:
        """
        Parse the model's response to extract entity list.
        """
        response = response.strip()

        response = re.sub(r"^```json\s*", "", response)
        response = re.sub(r"^```\s*", "", response)
        response = re.sub(r"\s*```$", "", response)

        try:
            entities = json.loads(response)
            if isinstance(entities, list):
                return [str(e).strip() for e in entities if e]
        except json.JSONDecodeError:
            pass

        match = re.search(r"\[([^\]]+)\]", response)
        if match:
            try:
                entities = json.loads(f"[{match.group(1)}]")
                if isinstance(entities, list):
                    return [str(e).strip() for e in entities if e]
            except json.JSONDecodeError:
                pass

        entities = re.split(r"[,\n]", response)
        entities = [
            e.strip().strip("\"'[]")
            for e in entities
            if e.strip() and not e.strip().startswith("{")
        ]

        return entities if entities else []
