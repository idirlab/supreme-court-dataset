import argparse
import pandas as pd
import json
import itertools
from tqdm import tqdm
import re

PROMPT_TEMPLATE = """You are a legal expert. Read two short claims about a case plus the case facts, question, and conclusion. Decide whether the two claims are saying the same thing (being redundant) or are meaningfully different regarding their meaning.
You must output an explanation for your decision in the "explanation" field. Then, also provide a decision in the "overlap" field: "redundant" if the claims are redundant, and "different" if they are not.

Claim 1: {claim1}
Claim 2: {claim2}

Case Evidence:
Facts: {facts}
Question: {question}
Conclusion: {conclusion}

## Output Format:
Return a JSON object in the following format:
```json
{{
    "explanation": "...",
    "overlap": "<redundant/different>",
    ...
}}
```
"""


def create_openai_message(prompt):
    return {"body": {"messages": [{"role": "user", "content": prompt}]}}

def generate_prompts(claims_file, metadata_file, output_prompts, output_metadata):
    print(f"Loading claims from {claims_file}...")
    claims_df = pd.read_csv(claims_file)
    
    print(f"Loading metadata from {metadata_file}...")
    meta_df = pd.read_csv(metadata_file)
    
    # Create a lookup for metadata
    # Assuming 'name' is unique or we take the first match
    meta_lookup = {}
    for _, row in meta_df.iterrows():
        if pd.notna(row['name']):
            meta_lookup[row['name']] = {
                'facts': row.get('facts', ''),
                'question': row.get('api_question', ''),
                'conclusion': row.get('api_conclusion', '')
            }
            
    prompts = []
    metadata_records = []
    
    # Group claims by case name
    grouped = claims_df.groupby('name')
    
    print("Generating pairs...")
    for name, group in tqdm(grouped):
        if name not in meta_lookup:
            # Try case-insensitive match or partial? 
            # For now, skip if not found, but print warning if needed.
            # print(f"Warning: Metadata not found for case '{name}'")
            continue
            
        case_meta = meta_lookup[name]
        claims = group['claim'].tolist()
        
        # Generate unique pairs
        for c1, c2 in itertools.combinations(claims, 2):
            # Skip if claims are identical strings (obviously overlap)
            if c1 == c2:
                continue
                
            prompt_text = PROMPT_TEMPLATE.format(
                facts=case_meta['facts'],
                question=case_meta['question'],
                conclusion=case_meta['conclusion'],
                claim1=c1,
                claim2=c2
            )
            
            prompts.append(create_openai_message(prompt_text))
            metadata_records.append({
                "case_name": name,
                "claim1": c1,
                "claim2": c2
            })
            
    print(f"Writing {len(prompts)} prompts to {output_prompts}...")
    with open(output_prompts, 'w') as f:
        for p in prompts:
            f.write(json.dumps(p) + '\n')
            
    print(f"Writing metadata to {output_metadata}...")
    with open(output_metadata, 'w') as f:
        for m in metadata_records:
            f.write(json.dumps(m) + '\n')
            
    print("Done.")

def extract_json(text):
    try:
        # Try to find JSON block
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        
        # Try to find just the object
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
            
        return None
    except:
        return None

def process_results(results_file, metadata_file, output_csv):
    print(f"Loading results from {results_file}...")
    results = []
    with open(results_file, 'r') as f:
        for line in f:
            results.append(json.loads(line))
            
    print(f"Loading metadata from {metadata_file}...")
    metadata = []
    with open(metadata_file, 'r') as f:
        for line in f:
            metadata.append(json.loads(line))
            
    if len(results) != len(metadata):
        print(f"Warning: Number of results ({len(results)}) does not match metadata ({len(metadata)}).")
        # We will proceed with the minimum length
        min_len = min(len(results), len(metadata))
        results = results[:min_len]
        metadata = metadata[:min_len]
        
    overlapping_pairs = []
    
    print("Processing results...")
    for res, meta in zip(results, metadata):
        # Extract output text
        # vLLM output format usually: {"outputs": [{"text": "..."}]} or similar
        # Or if it's the OpenAI format response
        output_text = ""
        if "outputs" in res and isinstance(res["outputs"], list):
            output_text = res["outputs"][0].get("text", "")
        elif "choices" in res and isinstance(res["choices"], list): # OpenAI format
             output_text = res["choices"][0].get("message", {}).get("content", "")
        else:
            # Fallback, maybe it's just the text?
            output_text = str(res)

        parsed = extract_json(output_text)
        
        decision = False
        explanation = ""
        
        if parsed:
            decision = parsed.get("overlap", False)
            explanation = parsed.get("explanation", "")
            
            # Handle string response based on prompt instructions ("redundant" vs "different")
            if isinstance(decision, str):
                decision_lower = decision.lower()
                # Check for "redundant" as requested in prompt, or "true" as fallback
                decision = "redundant" in decision_lower or "true" == decision_lower
        else:
            explanation = f"Failed to parse JSON. Raw output: {output_text[:100]}..."
            
        if decision:
            overlapping_pairs.append({
                "case_name": meta["case_name"],
                "claim_1": meta["claim1"],
                "claim_2": meta["claim2"],
                "decision": decision,
                "explanation": explanation
            })
            
    df = pd.DataFrame(overlapping_pairs)
    print(f"Found {len(df)} overlapping pairs.")
    df.to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    
    gen_parser = subparsers.add_parser("generate")
    gen_parser.add_argument("--claims-file", default="results_80B_claims.csv")
    gen_parser.add_argument("--metadata-file", default="clean_data_with_details.csv")
    gen_parser.add_argument("--output-prompts", default="overlap_prompts.jsonl")
    gen_parser.add_argument("--output-metadata", default="overlap_test_set.jsonl")

    proc_parser = subparsers.add_parser("process")
    proc_parser.add_argument("--results-file", required=True)
    proc_parser.add_argument("--metadata-file", required=True)
    proc_parser.add_argument("--output-csv", default="overlapping_claims.csv")
    
    args = parser.parse_args()
    
    if args.mode == "generate":
        generate_prompts(args.claims_file, args.metadata_file, args.output_prompts, args.output_metadata)
    elif args.mode == "process":
        process_results(args.results_file, args.metadata_file, args.output_csv)
