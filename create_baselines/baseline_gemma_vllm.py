import os
import json
import base64
from io import BytesIO
from typing import Dict, Any, Optional, List
from PIL import Image
from pydantic import BaseModel, Field
from tqdm import tqdm

from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

class HallucinationBaseline(BaseModel):
    objects: str = Field(description="Semantic description of object hallucinations.")
    background: str = Field(description="Semantic description of background hallucinations.")
    object_omission: str = Field(description="Semantic description of object omission hallucinations.")

# PROMPT TEMPLATE (Optimized for Gemma 3)
BASE_PROMPT_TEMPLATE = """"You are an image re-contextualization hallucination inspector.
You will be given:
1. An instruction prompt (what the generator was asked to do).
2. A background image (image 1).
3. Reference object image(s) (image 2, optionally image 3).
4. The generated image (last image) - result of re-contextualization.

Your task: compare the generated image to the references and to the instruction. For each category, report ONLY detected problems/hallucinations.

CRITICAL RULES:
- If a category has problems: describe them briefly (1-3 sentences).
- If a category has NO problems: return EXACTLY "" (empty string). Do NOT write "no issues", "correct", "accurate" or any description of correctness.

Categories:
1) objects: Object Visual Fidelity - texture/shape/color identity mismatches, mutations, identity loss, reference bleeding. Example: "Feature Mutation: sofa's color changed from dark green to black."
2) background: Background Fidelity - background mutations, background detail loss, context swap. Example: "Background Mutation: wall color changed."
3) object_omission: Object Omission - missing required objects from the instruction that should have been pasted from object image. Example: "Object Omission: green cabinet missing."

Example output if only objects have issues:
{{"objects": "Color mismatch: car is blue instead of red.", "background": "", "object_omission": ""}}

Instruction Prompt:
{instruction_prompt}

Analyze the provided images now.
Return only a JSON object with keys: objects, background, object_omission."""

def image_to_base64(image_path: str) -> str:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"

def load_existing_results(output_path: str):
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            processed = {r["generated_photo"] for r in results if "hallucination" in r}
            return results, processed
        except Exception: pass
    return [], set()

def process_dataset(dataset_info, llm):
    input_file = dataset_info["input"]
    output_file = dataset_info["output"]
    data_dir = dataset_info["data_dir"]

    results, processed_photos = load_existing_results(output_file)
    with open(input_file, "r", encoding="utf-8") as f:
        all_annotations = json.load(f)

    to_process = [a for a in all_annotations if a["generated_photo"] not in processed_photos]
    
    if not to_process:
        print(f"Category {os.path.basename(input_file)} already processed.")
        return

    sampling_params = SamplingParams(
        max_tokens=512,
        temperature=0,
        structured_outputs=StructuredOutputsParams(json=HallucinationBaseline.model_json_schema())
    )

    batch_size = 2 
    for i in tqdm(range(0, len(to_process), batch_size), desc=f"Gemma-3 Eval: {os.path.basename(input_file)}"):
        batch = to_process[i:i+batch_size]
        prompts = []
        valid_items = []

        for item in batch:
            photo_name = item["generated_photo"]
            example_id = photo_name[:4]
            base_path = os.path.join(data_dir, example_id)
            
            p_path = os.path.join(base_path, f"{example_id}prompt.txt")
            if not os.path.exists(p_path): continue

            with open(p_path, "r", encoding="utf-8") as f:
                prompt_text = f.read().strip()

            content = []
            for suffix in ["background.png", "object1.png", "object2.png", "generated.png"]:
                img_path = os.path.join(base_path, f"{example_id}{suffix}")
                if os.path.exists(img_path):
                    content.append({"type": "image_url", "image_url": {"url": image_to_base64(img_path)}})
            
            content.append({"type": "text", "text": BASE_PROMPT_TEMPLATE.format(instruction_prompt=prompt_text)})

            prompts.append([{"role": "user", "content": content}])
            valid_items.append(item)

        if not prompts: continue

        outputs = llm.chat(prompts, sampling_params=sampling_params)
        
        for item, out in zip(valid_items, outputs):
            try:
                new_h = json.loads(out.outputs[0].text)
                item["hallucination"] = new_h
                results.append(item)
            except: continue

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

def main():
    llm = LLM(
        model="google/gemma-3-12b-it",
        trust_remote_code=True,
        max_model_len=16384, 
        limit_mm_per_prompt={"image": 4},
    )

    DATASETS = [
        {"input": "", 
         "data_dir": "", 
         "output": ""},
        {"input": "", 
         "data_dir": "", 
         "output": ""},
        {"input": "", 
         "data_dir": "", 
         "output": ""},
        {"input": "", 
         "data_dir": "", 
         "output": ""},
        {"input": "", 
         "data_dir": "", 
         "output": ""},
    ]

    for ds in DATASETS:
        process_dataset(ds, llm)

if __name__ == "__main__":
    main()