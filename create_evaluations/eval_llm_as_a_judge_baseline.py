import json
import argparse
import os
import re
import time
from pathlib import Path
from collections import defaultdict
import pandas as pd
import google.generativeai as genai
from tqdm import tqdm

CATEGORIES = ['cars', 'clothes', 'consumer_goods', 'electronics', 'furniture']
HALLUCINATION_TYPES = ['objects', 'background', 'object_omission']

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

def calculate_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {'F1-score': round(f1, 4), 'Precision': round(precision, 4), 'Recall': round(recall, 4)}

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
                generation_config={"temperature": 0.0 if attempt == 0 else 0.2, "response_mime_type": "application/json"}
            )
            raw_text = response.text.strip()
            clean_json = re.sub(r"^```json\s*|^```\s*|```$", "", raw_text, flags=re.MULTILINE).strip()
            return json.loads(clean_json)
        except Exception as e:
            if attempt < retries:
                time.sleep(2)
                continue
            else:
                print(f"\n[!] Error after {retries} attempts (ID: {items[0]['id']}): {e}")
                return [{'id': it['id'], 'tp': 0, 'fn': 0, 'fp': 0} for it in items]

def run_evaluation(baseline_name, data_dir, batch_size):
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
    model = genai.GenerativeModel('gemini-2.5-flash')
    all_cat_stats = {}
    
    for cat in CATEGORIES:
        gt_f = data_dir / cat / f"annotations_{cat}_final.json"
        pred_f = data_dir / cat / f"annotations_{cat}_baseline_{baseline_name}.json"
        if not gt_f.exists() or not pred_f.exists(): continue
        
        with open(gt_f) as f: gt_data = json.load(f)
        with open(pred_f) as f: pred_data = {item['generated_photo']: item['hallucination'] for item in json.load(f)}
        
        cat_counts = {ht: {'tp': 0, 'fp': 0, 'fn': 0} for ht in HALLUCINATION_TYPES}
        queue = []
        for idx, gt_item in enumerate(gt_data):
            photo = gt_item['generated_photo']
            if photo not in pred_data: continue
            for ht in HALLUCINATION_TYPES:
                gt_txt = gt_item['hallucination'].get(ht, '')
                pred_txt = pred_data[photo].get(ht, '')
                if gt_txt or pred_txt:
                    queue.append({'id': f"{cat}_{idx}_{ht}", 'hallucination_type': ht, 'gt_text': gt_txt, 'pred_text': pred_txt, 'ht_key': ht})

        for i in tqdm(range(0, len(queue), batch_size), desc=f"Judging {cat.upper()}"):
            batch = queue[i:i+batch_size]
            results = llm_count_errors_batch(model, batch)
            res_dict = {str(r['id']): r for r in results if 'id' in r}
            for it in batch:
                res = res_dict.get(it['id'], {'tp': 0, 'fp': 0, 'fn': 0})
                cat_counts[it['ht_key']]['tp'] += int(res.get('tp', 0))
                cat_counts[it['ht_key']]['fp'] += int(res.get('fp', 0))
                cat_counts[it['ht_key']]['fn'] += int(res.get('fn', 0))
        all_cat_stats[cat] = cat_counts
    return all_cat_stats

def generate_text_report(stats, baseline_name):
    cat_metrics = {}
    for cat, s in stats.items():
        f1_list = []
        for ht in HALLUCINATION_TYPES:
            f1_list.append(calculate_metrics(s[ht]['tp'], s[ht]['fp'], s[ht]['fn'])['F1-score'])
        cat_metrics[cat] = {'macro_f1': sum(f1_list) / 3}

    lines = [f"\n{'='*100}", f"  FINAL LLM-AS-A-JUDGE REPORT: {baseline_name.upper()}", f"{'='*100}\n"]
    all_macro = sum(m['macro_f1'] for m in cat_metrics.values()) / len(cat_metrics)
    lines.append(f"OVERALL PERFORMANCE (Macro All 5): {all_macro:.4f}\n{'-'*100}\n")
    lines.append(f"{'CATEGORY':<25} | {'ONLY CAT (Macro)':<20} | {'4-EXCL CAT (Stability)':<20}\n{'-'*100}")
    for cat in CATEGORIES:
        if cat in cat_metrics:
            others = [c for c in CATEGORIES if c != cat and c in cat_metrics]
            excl_f1 = sum(cat_metrics[o]['macro_f1'] for o in others) / len(others) if others else 0
            lines.append(f"{cat.upper():<25} | {cat_metrics[cat]['macro_f1']:<20.4f} | {excl_f1:<20.4f}")
    return "\n".join(lines + ["="*100])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=15)
    args = parser.parse_args()

    DATA_DIR = Path("./data")
    stats = run_evaluation(args.baseline, DATA_DIR, args.batch_size)
    report = generate_text_report(stats, args.baseline)
    
    with open(f"JUDGE_REPORT_{args.baseline}.txt", "w") as f: f.write(report)
    print(report)