import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

CATEGORIES = ['cars', 'clothes', 'consumer_goods', 'electronics', 'furniture']
HALLUCINATION_TYPES = ['objects', 'background', 'object_omission']
DATA_DIR = Path("")
OUTPUTS_BASE = Path("")
GT_SUFFIX = ""
OUTPUT_CSV = ""

def calculate_all_metrics(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    return prec, rec, f1

def run_report():
    experiments = [d for d in OUTPUTS_BASE.iterdir() if d.is_dir()]
    rows = []

    print(f"Starting analysis of {len(experiments)} experiments in 4+1 format...")

    for exp_path in tqdm(experiments, desc="Experiments"):
        exp_name = exp_path.name
        cat_results = {}
        
        for cat in CATEGORIES:
            gt_f = DATA_DIR / cat / f'annotations_{cat}_{GT_SUFFIX}.json'
            pipe_f = DATA_DIR / cat / f'annotations_{cat}_{exp_name}.json'
            
            if not gt_f.exists() or not pipe_f.exists():
                continue

            with open(gt_f, 'r') as f: gt_d = json.load(f)
            with open(pipe_f, 'r') as f: pipe_d = json.load(f)

            pipe_map = {item['generated_photo']: item['hallucination'] for item in pipe_d}
            
            type_metrics = []
            for ht in HALLUCINATION_TYPES:
                tp = fp = fn = 0
                for g_item in gt_d:
                    photo = g_item['generated_photo']
                    gh = bool(g_item['hallucination'].get(ht))
                    ph = bool(pipe_map.get(photo, {}).get(ht))
                    
                    if ph and gh: tp += 1
                    elif ph and not gh: fp += 1
                    elif not ph and gh: fn += 1
                
                type_metrics.append(calculate_all_metrics(tp, fp, fn))
            
            cat_results[cat] = {
                'p': sum(m[0] for m in type_metrics) / 3,
                'r': sum(m[1] for m in type_metrics) / 3,
                'f1': sum(m[2] for m in type_metrics) / 3
            }

        if len(cat_results) < len(CATEGORIES):
            continue 

        for excluded_cat in CATEGORIES:
            others = [cat_results[c] for c in CATEGORIES if c != excluded_cat]
            group4_f1 = sum(c['f1'] for c in others) / 4
            group4_prec = sum(c['p'] for c in others) / 4
            group4_rec = sum(c['r'] for c in others) / 4
            
            only1 = cat_results[excluded_cat]
            
            rows.append({
                'Experiment': exp_name,
                'Excluded_Category': excluded_cat,
                'Group_4_Name': f"All_except_{excluded_cat}",
                'Group4_F1': group4_f1,
                'Group4_Prec': group4_prec,
                'Group4_Rec': group4_rec,
                'Only1_Name': excluded_cat,
                'Only1_F1': only1['f1'],
                'Only1_Prec': only1['p'],
                'Only1_Rec': only1['r']
            })

    df = pd.DataFrame(rows)

    df = df.sort_values(by=['Excluded_Category', 'Group4_F1'], ascending=[True, False])

    df.to_csv(OUTPUT_CSV, index=False, sep=';', encoding='utf-8-sig')
    
    print(f"Report 4+1 saved to: {OUTPUT_CSV}")
    
    example_cat = CATEGORIES[0]
    print(f"\nTop 5 experiments for 4 categories (excluding {example_cat}):")
    print(df[df['Excluded_Category'] == example_cat].head(5)[['Experiment', 'Group4_F1', 'Only1_F1']])

if __name__ == "__main__":
    run_report()