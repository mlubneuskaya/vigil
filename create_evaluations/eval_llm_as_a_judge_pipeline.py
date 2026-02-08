import json
import argparse
import os
import re
import time
from pathlib import Path
import pandas as pd
import google.generativeai as genai
from tqdm import tqdm

JUDGE_PROMPT_TEMPLATE = """You are an expert auditor evaluating AI hallucination detection.
    
Hallucination Type: {hallucination_type}

Human Ground Truth (list of real errors): 
"{ground_truth_text}"

AI Pipeline Output (list of detected errors): 
"{pipeline_text}"

Task:
Break down BOTH descriptions into individual distinct errors/issues and count:

1. TP (True Positives): How many specific errors from Ground Truth did the AI correctly detect?
   - Synonyms count as matches (e.g., "missing leg" == "leg not visible")
   - Similar descriptions of the same error count as TP

2. FN (False Negatives): How many specific errors from Ground Truth did the AI MISS completely?
   - Count each distinct error that's in Ground Truth but NOT in AI output

3. FP (False Positives): How many NEW errors did the AI report that are NOT in Ground Truth?
   - Count each distinct error that's in AI output but NOT in Ground Truth

Rules:
- Treat each distinct object/issue as a separate error instance
- If multiple errors are described in one sentence, count them separately
- Be precise and conservative in counting

Examples:
- Ground Truth ="Object omission: missing red car and blue truck" has 2 errors, not 1
- Ground Truth ="Object mutation: the back and sides of the car do not correspond to the reference" has 1 error
- For instance:
    - "Object mutation: The jeans have different details"
    - "Object mutation: The jeans show incorrect stitching patterns and colors"
    Both descriptions refer to the same visual inconsistency and should be treated as a single TP.

Output ONLY valid JSON in this exact format:
{{
    "tp": <integer>,
    "fn": <integer>,
    "fp": <integer>
}}
"""

CATEGORIES = ['cars', 'clothes', 'consumer_goods', 'electronics', 'furniture']
HALLUCINATION_TYPES = ['objects', 'background', 'object_omission']

def calculate_metrics(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
    return {'f1': round(f1, 4), 'p': round(p, 4), 'r': round(r, 4)}

def llm_count_errors_batch(model, items, retries=2):
    if not items: return []
    
    batch_tasks = []
    for it in items:
        gt_display = it['gt_text'] if it['gt_text'] else '[EMPTY - no errors]'
        pred_display = it['pred_text'] if it['pred_text'] else '[EMPTY - no errors detected]'
        task_str = JUDGE_PROMPT_TEMPLATE.format(
            hallucination_type=it['hallucination_type'],
            ground_truth_text=gt_display,
            pipeline_text=pred_display
        )
        batch_tasks.append(f"CASE ID: {it['id']}\n{task_str}\n---")

    final_prompt = (
        "Respond with a JSON array of objects. Each object MUST have: \"id\" (string), \"tp\" (int), \"fn\" (int), \"fp\" (int). "
        "Do not include any reasoning or extra text.\n\n" + "\n".join(batch_tasks)
    )

    for attempt in range(retries + 1):
        try:
            response = model.generate_content(
                final_prompt, 
                generation_config={"temperature": 0.0, "response_mime_type": "application/json"}
            )
            raw_text = response.text.strip()
            clean_json = re.sub(r"^```json\s*|^```\s*|```$", "", raw_text, flags=re.MULTILINE).strip()
            return json.loads(clean_json)
        except Exception as e:
            if attempt < retries:
                time.sleep(3)
                continue
            else:
                return [{'id': it['id'], 'tp': 0, 'fn': 0, 'fp': 0} for it in items]

def run_eval():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suffix', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=20)
    args = parser.parse_args()

    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    cat_results = {}
    detailed_rows = []
    data_dir = Path("./data")

    for cat in CATEGORIES:
        gt_f = data_dir / cat / f"annotations_{cat}_final.json"
        pi_f = data_dir / cat / f"annotations_{cat}_{args.suffix}.json"
        
        if not gt_f.exists() or not pi_f.exists(): continue
        
        with open(gt_f) as f: gt_data = json.load(f)
        with open(pi_f) as f: pred_data = {it['generated_photo']: it['hallucination'] for it in json.load(f)}
        
        queue = []
        for idx, gt_it in enumerate(gt_data):
            photo = gt_it['generated_photo']
            if photo not in pred_data: continue
            for ht in HALLUCINATION_TYPES:
                gt_txt = gt_it['hallucination'].get(ht, '')
                pred_txt = pred_data[photo].get(ht, '')
                if gt_txt or pred_txt:
                    queue.append({
                        'id': f"{cat}_{idx}_{ht}", 
                        'hallucination_type': ht, 
                        'gt_text': gt_txt, 
                        'pred_text': pred_txt, 
                        'ht_key': ht
                    })

        counts = {ht: {'tp': 0, 'fp': 0, 'fn': 0} for ht in HALLUCINATION_TYPES}
        for i in tqdm(range(0, len(queue), args.batch_size), desc=f"Eval {cat.upper()}"):
            batch = queue[i:i+args.batch_size]
            results = llm_count_errors_batch(model, batch)
            
            res_dict = {str(r['id']): r for r in results if 'id' in r}
            for it in batch:
                res = res_dict.get(it['id'], {'tp': 0, 'fp': 0, 'fn': 0})
                counts[it['ht_key']]['tp'] += int(res.get('tp', 0))
                counts[it['ht_key']]['fp'] += int(res.get('fp', 0))
                counts[it['ht_key']]['fn'] += int(res.get('fn', 0))
                detailed_rows.append({**it, **res, 'cat': cat})
        
        # Obliczanie metryk dla kategorii
        m_list = [calculate_metrics(counts[ht]['tp'], counts[ht]['fp'], counts[ht]['fn']) for ht in HALLUCINATION_TYPES]
        cat_results[cat] = {
            'f1': sum(m['f1'] for m in m_list)/3,
            'p': sum(m['p'] for m in m_list)/3,
            'r': sum(m['r'] for m in m_list)/3
        }

    # REPORT TXT 
    report_path = f"REPORT_PIPELINE_NEW_{args.suffix}.txt"
    with open(report_path, "w") as f:
        f.write(f"PIPELINE REPORT: {args.suffix}\n{'='*100}\n")
        all_f1 = sum(v['f1'] for v in cat_results.values()) / len(cat_results) if cat_results else 0
        f.write(f"OVERALL MACRO F1: {all_f1:.4f}\n{'-'*100}\n")
        f.write(f"{'CATEGORY':<20} | {'PREC':<8} | {'RECALL':<8} | {'ONLY CAT':<12} | {'4-EXCL CAT':<12}\n{'-'*100}\n")
        
        for cat in CATEGORIES:
            if cat in cat_results:
                others = [cat_results[c]['f1'] for c in CATEGORIES if c != cat and c in cat_results]
                excl_f1 = sum(others)/len(others) if others else 0
                r = cat_results[cat]
                f.write(f"{cat.upper():<20} | {r['p']:<8.4f} | {r['r']:<8.4f} | {r['f1']:<12.4f} | {excl_f1:<12.4f}\n")

    # Save details to csv
    pd.DataFrame(detailed_rows).to_csv(f"DETAILS_PIPELINE_NEW_{args.suffix}.csv", index=False)

if __name__ == "__main__":
    run_eval()
