import argparse
import pandas as pd
import json
from tqdm import tqdm
import re
import random

PROMPT_TEMPLATE_SINGLE = """You are a legal expert. Read the following claim about a legal case, along with the case's facts, question, and conclusion.
Your task is to generate a negation of this claim. The negation should be plausible but factually incorrect based on the original claim and the case evidence.
Provide an explanation for why this negation contradicts the original claim in the "explanation" field. Then, provide the negated claim in the "negation" field.

Claim: {claim}

Case Evidence:
Facts: {facts}
Question: {question}
Conclusion: {conclusion}

## Output Format:
Return a JSON object in the following format:
```json
{{
    "explanation": "...",
    "negation": "..."
}}
```
"""

PROMPT_TEMPLATE_MULTI = """You are a legal expert. Read the following claim about a legal issue, along with evidence from multiple relevant cases.
Your task is to generate a negation of this claim. The negation should be plausible but factually incorrect based on the original claim and the provided case evidence.
Provide an explanation for why this negation contradicts the original claim in the "explanation" field. Then, provide the negated claim in the "negation" field.

Claim: {claim}

Case Evidence:
{cases_evidence}

## Output Format:
Return a JSON object in the following format:
```json
{{
    "explanation": "...",
    "negation": "..."
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
    
    print("Generating prompts...")
    for _, row in tqdm(claims_df.iterrows(), total=len(claims_df)):
        claim = row['claim']
        fact_id = row['fact_id']
        claim_type = str(row.get('type', 'none')).strip()
        
        prompt_text = ""
        case_name_used = ""
        
        # Default empty meta
        default_meta = {'facts': 'N/A', 'question': 'N/A', 'conclusion': 'N/A'}
        
        if claim_type == 'overruling':
            # Use overruling_case
            c_name = str(row.get('overruling_case', '')).split(';')[0].strip()
            case_name_used = c_name
            
            case_meta = meta_lookup.get(c_name, default_meta)
            prompt_text = PROMPT_TEMPLATE_SINGLE.format(
                claim=claim,
                facts=case_meta['facts'],
                question=case_meta['question'],
                conclusion=case_meta['conclusion']
            )
            
        elif claim_type == 'confirmed':
            # Use case_name list
            try:
                cases = json.loads(row.get('case_name', '[]'))
            except:
                cases = []
                
            if not isinstance(cases, list):
                cases = [str(row.get('case_name', ''))]
            
            # Filter empty
            cases = [c for c in cases if c]
            
            if len(cases) <= 1:
                # Treat as single
                c_name = cases[0] if cases else ""
                case_name_used = c_name
                case_meta = meta_lookup.get(c_name, default_meta)
                prompt_text = PROMPT_TEMPLATE_SINGLE.format(
                    claim=claim,
                    facts=case_meta['facts'],
                    question=case_meta['question'],
                    conclusion=case_meta['conclusion']
                )
            else:
                # Multiple cases
                if len(cases) > 3:
                    selected_cases = random.sample(cases, 3)
                else:
                    selected_cases = cases
                
                case_name_used = json.dumps(selected_cases)
                
                evidence_parts = []
                for c_name in selected_cases:
                    meta = meta_lookup.get(c_name, default_meta)
                    evidence_parts.append(f"Case: {c_name}\nFacts: {meta['facts']}\nQuestion: {meta['question']}\nConclusion: {meta['conclusion']}")
                
                cases_evidence = "\n\n".join(evidence_parts)
                prompt_text = PROMPT_TEMPLATE_MULTI.format(claim=claim, cases_evidence=cases_evidence)
                
        else: # 'none' or others
            # Use first case in case_name
            try:
                cases = json.loads(row.get('case_name', '[]'))
                if isinstance(cases, list) and len(cases) > 0:
                    c_name = cases[0]
                else:
                    c_name = str(row.get('case_name', ''))
            except:
                c_name = str(row.get('case_name', ''))
            
            # Clean up if it looks like a list string but failed json load
            if c_name.startswith("['") and c_name.endswith("']"):
                 c_name = c_name[2:-2]
            
            case_name_used = c_name
            case_meta = meta_lookup.get(c_name, default_meta)
            prompt_text = PROMPT_TEMPLATE_SINGLE.format(
                claim=claim,
                facts=case_meta['facts'],
                question=case_meta['question'],
                conclusion=case_meta['conclusion']
            )
        
        prompts.append(create_openai_message(prompt_text))
        metadata_records.append({
            "fact_id": fact_id,
            "original_claim": claim,
            "case_name_used": case_name_used,
            "type": claim_type,
            "judgement": row.get('judgement', ''),
            "explanation": row.get('explanation', ''),
            "overruling_case": row.get('overruling_case', '')
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

def process_results(results_file, claims_file, output_csv):
    print(f"Loading claims from {claims_file}...")
    claims_df = pd.read_csv(claims_file)
    
    # Create a lookup: claim -> row data
    # We need to map back to the original row to get all columns
    claim_lookup = {}
    for _, row in claims_df.iterrows():
        if pd.notna(row['claim']):
            claim_lookup[row['claim'].strip()] = row.to_dict()
            
    print(f"Loading results from {results_file}...")
    results = []
    with open(results_file, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
                
    final_rows = []
    
    print("Processing results...")
    for res in tqdm(results):
        # Extract prompt
        prompt = res.get("prompt", "")
        if not prompt:
            continue
            
        # Extract claim from prompt
        # Pattern: Claim: {claim}\n
        # New format: Claim: {claim}\n\nCase Evidence:
        match = re.search(r"Claim: (.*?)\n\nCase Evidence:", prompt, re.DOTALL)
        if not match:
             # Fallback to old pattern if needed
             match = re.search(r"Claim: (.*?)\n", prompt)
        
        if not match:
            continue
            
        original_claim_text = match.group(1).strip()
        
        # Lookup original data
        original_data = claim_lookup.get(original_claim_text)
        if not original_data:
            # Try to find by partial match or just skip?
            # For now skip
            continue
        
        # Extract answer
        output_text = ""
        if "answer" in res:
            output_text = res["answer"]
        elif "outputs" in res and isinstance(res["outputs"], list):
            output_text = res["outputs"][0].get("text", "")
        elif "choices" in res and isinstance(res["choices"], list):
            output_text = res["choices"][0].get("message", {}).get("content", "")
        
        parsed = extract_json(output_text)
        negation = ""
        
        if parsed:
            negation = parsed.get("negation", "")
        
        if not negation:
            # Fallback or skip?
            # If we can't parse, we might lose this negative sample.
            # Let's log it and skip for now.
            continue
            
        # Create Positive Sample
        pos_row = original_data.copy()
        pos_row['label'] = 1
        pos_row['is_negation'] = False
        final_rows.append(pos_row)
        
        # Create Negative Sample
        neg_row = original_data.copy()
        neg_row['claim'] = negation
        neg_row['label'] = 0
        neg_row['is_negation'] = True
        # Clear fields that might not apply to negation or keep them?
        # The user said "Have a shared 'fact_id' for each positive-negative pair"
        # So we keep fact_id.
        # Other fields like 'judgement', 'explanation' refer to the case/original claim relation.
        # For the negative claim, the judgement would technically be opposite, but we are just creating a dataset.
        # I'll keep the metadata but the claim is different.
        final_rows.append(neg_row)
            
    df = pd.DataFrame(final_rows)
    print(f"Generated {len(df)} rows (original + negations).")
    
    # Sort by fact_id to keep pairs together
    if 'fact_id' in df.columns:
        df = df.sort_values('fact_id')
        
    df.to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    
    gen_parser = subparsers.add_parser("generate")
    gen_parser.add_argument("--claims-file", default="../diff_case_claims_updated.csv")
    gen_parser.add_argument("--metadata-file", default="../clean_data_with_details.csv")
    gen_parser.add_argument("--output-prompts", default="../negation_prompts.jsonl")
    gen_parser.add_argument("--output-metadata", default="../negation_metadata.jsonl")

    proc_parser = subparsers.add_parser("process")
    proc_parser.add_argument("--results-file", default="../negation_output.jsonl")
    proc_parser.add_argument("--claims-file", default="../diff_case_claims_updated.csv")
    proc_parser.add_argument("--output-csv", default="../diff_case_claims_with_negatives.csv")
    
    args = parser.parse_args()
    
    if args.mode == "generate":
        generate_prompts(args.claims_file, args.metadata_file, args.output_prompts, args.output_metadata)
    elif args.mode == "process":
        process_results(args.results_file, args.claims_file, args.output_csv)
