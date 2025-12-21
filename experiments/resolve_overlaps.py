import argparse
import pandas as pd
import json
import re
from tqdm import tqdm
from pathlib import Path

PROMPT_TEMPLATE = """You are a legal expert. You are given two claims about the same Supreme Court case that are redundant (saying the same thing). Read the two short claims about the case and the case facts, question, and conclusion. 
Your task is to choose the one that is better written, more precise, or more comprehensive. You are also given the reason for redundancy, take that into account when making your decision. In the "keep" json field, return the strings "claim1" or "claim2", not the actual content of the claims.

Claim 1: {claim1}
Claim 2: {claim2}

Reason for redundancy: {explanation}

Case Evidence:
Facts: {facts}
Question: {question}
Conclusion: {conclusion}

## Output Format:
Return a JSON object in the following format:
```json
{{
    "explanation": "...",
    "keep": "<claim1/claim2>"
}}
```
"""

def create_openai_message(prompt):
    return {"body": {"messages": [{"role": "user", "content": prompt}]}}

def load_csv(path):
    if Path(path).exists():
        return pd.read_csv(path)
    return None

def generate_prompts(overlaps_file, factual_claims_file, metadata_file, output_prompts, output_metadata):
    print(f"Loading overlaps from {overlaps_file}...")
    df_overlap = load_csv(overlaps_file)
    if df_overlap is None:
        print("Overlaps file not found.")
        return

    print(f"Loading factual claims from {factual_claims_file}...")
    df_factual = load_csv(factual_claims_file)
    if df_factual is None:
        print("Factual claims file not found.")
        return
    
    # Create a set of valid claims for fast lookup
    valid_claims = set(df_factual['claim'].astype(str).str.lower().str.strip())

    print(f"Loading metadata from {metadata_file}...")
    meta_df = load_csv(metadata_file)
    meta_lookup = {}
    if meta_df is not None:
        for _, row in meta_df.iterrows():
            if pd.notna(row['name']):
                meta_lookup[row['name']] = {
                    'facts': row.get('facts', ''),
                    'question': row.get('api_question', ''),
                    'conclusion': row.get('api_conclusion', '')
                }

    prompts = []
    metadata_records = []
    
    print("Generating prompts...")
    count = 0
    for _, row in tqdm(df_overlap.iterrows(), total=len(df_overlap)):
        if row['decision'] != 'redundant':
            continue

        c1 = str(row['claim_1']).strip()
        c2 = str(row['claim_2']).strip()
        case_name = row['case_name']
        explanation = row['explanation']

        # Check if both claims are in factual_claims
        if c1.lower() not in valid_claims or c2.lower() not in valid_claims:
            continue

        if case_name not in meta_lookup:
            continue

        case_meta = meta_lookup[case_name]
        
        prompt_text = PROMPT_TEMPLATE.format(
            case_name=case_name,
            facts=case_meta['facts'],
            question=case_meta['question'],
            conclusion=case_meta['conclusion'],
            claim1=c1,
            claim2=c2,
            explanation=explanation
        )
        
        prompts.append(create_openai_message(prompt_text))
        metadata_records.append({
            "case_name": case_name,
            "claim1": c1,
            "claim2": c2
        })
        count += 1

    print(f"Generated {count} prompts.")
    
    with open(output_prompts, 'w') as f:
        for p in prompts:
            f.write(json.dumps(p) + '\n')
            
    with open(output_metadata, 'w') as f:
        for m in metadata_records:
            f.write(json.dumps(m) + '\n')

def extract_json(text):
    try:
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        return None
    except:
        return None

def process_results(results_file, factual_claims_file, output_file):
    print(f"Loading results from {results_file}...")
    results = []
    with open(results_file, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    print(f"Loading factual claims from {factual_claims_file}...")
    df_factual = load_csv(factual_claims_file)
    if df_factual is None:
        return

    claims_to_remove = set()
    
    # Statistics
    stats = {
        "claim1_kept": 0,
        "claim2_kept": 0
    }

    print("Processing results...")
    for res in tqdm(results):
        # Extract prompt info to identify claims
        prompt = res.get("prompt", "")
        if not prompt:
            continue
            
        match = re.search(r"Claim 1:\s*(.*?)\nClaim 2:\s*(.*?)\n", prompt)
        if not match:
            continue
            
        c1 = match.group(1).strip()
        c2 = match.group(2).strip()
        
        # Extract answer
        output_text = ""
        if "answer" in res:
            output_text = res["answer"]
        elif "outputs" in res and isinstance(res["outputs"], list):
            output_text = res["outputs"][0].get("text", "")
        elif "choices" in res and isinstance(res["choices"], list):
            output_text = res["choices"][0].get("message", {}).get("content", "")
            
        parsed = extract_json(output_text)
        
        if parsed:
            keep = parsed.get("keep", "").lower()
            
            if "claim1" in keep:
                print(f"Removing Claim 2: {c2}")
                claims_to_remove.add(c2)
                stats["claim1_kept"] += 1
            elif "claim2" in keep:
                print(f"Removing Claim 1: {c1}")
                claims_to_remove.add(c1)
                stats["claim2_kept"] += 1

    # Apply changes
    print("\nStatistics:")
    print(f"  Claim 1 kept: {stats['claim1_kept']}")
    print(f"  Claim 2 kept: {stats['claim2_kept']}")
    print(f"Removing {len(claims_to_remove)} claims...")
    
    # Filter out removed claims
    # Normalize for comparison
    df_factual['claim_norm'] = df_factual['claim'].astype(str).str.strip()
    df_final = df_factual[~df_factual['claim_norm'].isin(claims_to_remove)].copy()
    df_final = df_final.drop(columns=['claim_norm'])
    
    print(f"Final claim count: {len(df_final)}")
    df_final.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    
    gen_parser = subparsers.add_parser("generate")
    gen_parser.add_argument("--overlaps-file", default="../results/overlapping_claims.csv")
    gen_parser.add_argument("--factual-claims-file", default="../results/factual_claims_resolved_contradictions.csv")
    gen_parser.add_argument("--metadata-file", default="../clean_data_with_details.csv")
    gen_parser.add_argument("--output-prompts", default="../overlap_resolution_prompts.jsonl")
    gen_parser.add_argument("--output-metadata", default="../overlap_resolution_metadata.jsonl")

    proc_parser = subparsers.add_parser("process")
    proc_parser.add_argument("--results-file", required=True)
    proc_parser.add_argument("--factual-claims-file", default="../results/factual_claims_resolved_contradictions.csv")
    proc_parser.add_argument("--output-file", default="../results/factual_claims_resolved_overlaps.csv")
    
    args = parser.parse_args()
    
    if args.mode == "generate":
        generate_prompts(args.overlaps_file, args.factual_claims_file, args.metadata_file, args.output_prompts, args.output_metadata)
    elif args.mode == "process":
        process_results(args.results_file, args.factual_claims_file, args.output_file)
