import os
import json
from typing import Dict, Any, Optional, List
from PIL import Image
from pydantic import BaseModel, Field
from tqdm import tqdm
import google.generativeai as genai  

class HallucinationBaseline(BaseModel):
    objects: str = Field(description="Semantic description of object hallucinations.")
    background: str = Field(description="Semantic description of background hallucinations.")
    object_omission: str = Field(description="Semantic description of object omission hallucinations.")

BASE_PROMPT_TEMPLATE = """
You are an image re-contextualization hallucination inspector.
You will be given:
- An instruction prompt (what the generator was asked to do).
- A background image (background_image).
- Reference object image(s) (one or two) - object1_image, object2_image.
- The generated image (generated_image) - result of re-contextualization.

Your task: compare the generated_image to the references and to the instruction, and produce THREE SHORT SEMANTIC DESCRIPTIONS (1-3 sentences each) answering the following categories. If no issues or hallucinations are detected in a category, return an empty string ("") for that category.

Categories:
1) objects: Object Visual Fidelity - texture/shape/color identity mismatches, mutations, identity loss, reference bleeding. Example: "Feature Mutation: sofa's color changed from dark green to black; Identity Loss: inserted cabinet is metallic vs wicker."
2) background: Background Fidelity - background mutations, background detail loss, context swap. Example: "Background Mutation: wall color changed; Context Swap: bedroom replaced by living room."
3) object_omission: Object Omission - missing required objects from the instruction that should have been pasted from object image. Example: "Object Omission: green cabinet missing."

Instruction Prompt:
{instruction_prompt}

Analyze the provided images now.
"""

def load_existing_results(output_path: str):
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            processed = {r["generated_photo"] for r in results if "hallucination" in r}
            return results, processed
        except Exception as e:
            print(f"  [!] Error reading {output_path}: {e}")
    return [], set()

def analyze_example(instruction_prompt: str, background_path: str, object1_path: str, generated_path: str, object2_path: Optional[str] = None):
    images = []
    try:
        images.append(Image.open(background_path))
        images.append(Image.open(object1_path))
        if object2_path:
            images.append(Image.open(object2_path))
        images.append(Image.open(generated_path))
    except Exception as e:
        return {"_error": f"File error: {e}"}

    text_prompt = BASE_PROMPT_TEMPLATE.format(instruction_prompt=instruction_prompt)
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(
            [text_prompt] + images,
            generation_config=genai.types.GenerationConfig(
                temperature=0,
                response_mime_type="application/json",
                response_schema=HallucinationBaseline,
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        return {"_error": f"Gemini error: {str(e)}"}

def process_dataset(dataset_info):
    input_file = dataset_info["input"]
    output_file = dataset_info["output"]
    data_dir = dataset_info["data_dir"]

    results, processed_photos = load_existing_results(output_file)
    
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            all_annotations = json.load(f)
    except FileNotFoundError:
        print(f"Input file not found: {input_file}")
        return

    to_process = [a for a in all_annotations if a["generated_photo"] not in processed_photos]
    
    print(f"\nCategory: {os.path.basename(input_file)}")
    print(f"  - Total in final: {len(all_annotations)}")
    print(f"  - Already processed: {len(processed_photos)}")
    print(f"  - To be processed: {len(to_process)}")
    
    if not to_process:
        print("  - Status: Kompletne.")
        return

    for annotation in tqdm(to_process, desc="Analysis Gemini"):
        photo_name = annotation["generated_photo"]
        example_id = photo_name[:4] 
        
        base_path = os.path.join(data_dir, example_id)
        prompt_path = os.path.join(base_path, f"{example_id}prompt.txt")
        
        if not os.path.exists(prompt_path):
            continue
            
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_text = f.read().strip()

        possible_gen_names = [
            f"{example_id}generated.png",
            f"{example_id}generated_01.png",
            photo_name 
        ]
        
        actual_gen_path = None
        for name in possible_gen_names:
            test_path = os.path.join(base_path, name)
            if os.path.exists(test_path):
                actual_gen_path = test_path
                break
        
        if not actual_gen_path:
            print(f"  [!] Skipped {photo_name}: Image file not found in {base_path}")
            continue
        # ----------------------------------------------

        # Analysis
        res = analyze_example(
            prompt_text,
            os.path.join(base_path, f"{example_id}background.png"),
            os.path.join(base_path, f"{example_id}object1.png"),
            actual_gen_path,
            os.path.join(base_path, f"{example_id}object2.png") if os.path.exists(os.path.join(base_path, f"{example_id}object2.png")) else None
        )

        if "_error" in res:
            print(f"  [!] Gemini API error for {photo_name}: {res['_error']}")
            continue

        results.append({**annotation, "hallucination": res})
        
        # Save after each record (safety for Gemini Rate Limits)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

def main():
    # complete paths before running
    DATASETS = [
        {
            "input": "",
            "data_dir": "",
            "output": "",
        },
        {
            "input": "",
            "data_dir": "",
            "output": "",
        },
        {
            "input": "",
            "data_dir": "",
            "output": "",
        },
        {
            "input": "",
            "data_dir": "",
            "output": "",
        },
        {
            "input": "",
            "data_dir": "",
            "output": "",
        },
    ]
    
    for ds in DATASETS:
        process_dataset(ds)

if __name__ == "__main__":
    main()
