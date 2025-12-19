import argparse
import pandas as pd
import json
from tqdm import tqdm
import re
import os

PROMPT_TEMPLATE = """You are a legal expert. Read a short claim about a supreme court case plus the case facts, question, and conclusion. Decide whether the claim is relevant for establishing a "legal rule" or principle derived from the case.

Rules:
1. A legal rule is a principle, regulation, or standard established by the court decision.
2. Determine if the claim discusses, cites, or relies on a legal rule from the case.
3. Base your judgment only on the provided claim and case evidence.
4. Ignore factual accuracy of the claim for this task; focus on whether it *attempts* to state a legal rule or implication.

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
    "is_legal_rule": "<yes/no>",
    "relevance_score": <int 1-5>
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
        # Adapt column names for politifact_sc_matches_scraped.csv
        # It has 'SC_Case_Match' and 'Claim'
        name = row.get('SC_Case_Match')
        claim = row.get('Claim')
        
        # Fallback to original names if not found (in case user uses different file)
        if pd.isna(name):
            name = row.get('name')
        if pd.isna(claim):
            claim = row.get('claim')
        
        if pd.isna(name) or pd.isna(claim):
            continue
            
        if name not in meta_lookup:
            # Try case insensitive match if direct match fails
            found = False
            for k in meta_lookup:
                if k.lower() == name.lower():
                    name = k
                    found = True
                    break
            if not found:
                continue
            
        case_meta = meta_lookup[name]
        
        prompt_text = PROMPT_TEMPLATE.format(
            claim=claim,
            facts=case_meta['facts'],
            question=case_meta['question'],
            conclusion=case_meta['conclusion']
        )
        
        prompts.append(create_openai_message(prompt_text))
        metadata_records.append({
            "case_name": name,
            "claim": claim,
            "original_id": row.get('Politifact_ID', '')
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
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        return None
    except:
        return None

def process_results(results_file, claims_file, output_csv):
    print(f"Loading claims from {claims_file}...")
    claims_df = pd.read_csv(claims_file)
    
    # Create a lookup: claim -> case_name
    # This might be ambiguous if same claim text maps to multiple cases, but let's assume unique enough or just take one
    claim_to_case = {}
    for _, row in claims_df.iterrows():
        c = row.get('Claim') or row.get('claim')
        n = row.get('SC_Case_Match') or row.get('name')
        if pd.notna(c) and pd.notna(n):
            claim_to_case[c.strip()] = n

    print(f"Loading results from {results_file}...")
    results = []
    with open(results_file, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
                
    processed_data = []
    
    print("Processing results...")
    for res in tqdm(results):
        # Extract prompt
        prompt = res.get("prompt", "")
        if not prompt:
            # Try to find prompt in messages if it's a chat format
            if "body" in res and "messages" in res["body"]:
                 prompt = res["body"]["messages"][0]["content"]
            
        if not prompt:
            continue
            
        # Extract claim from prompt
        # Pattern: Claim: ...\n\nCase Evidence:
        match = re.search(r"Claim:\s*(.*?)\s*Case Evidence:", prompt, re.DOTALL)
        if not match:
            continue
            
        claim = match.group(1).strip()
        
        # Lookup case name
        case_name = claim_to_case.get(claim, "Unknown")
        
        # Extract answer
        output_text = ""
        if "answer" in res:
            output_text = res["answer"]
        elif "outputs" in res and isinstance(res["outputs"], list):
            output_text = res["outputs"][0].get("text", "")
        elif "choices" in res and isinstance(res["choices"], list):
            output_text = res["choices"][0].get("message", {}).get("content", "")
        elif "response" in res: # Another possible format
             output_text = res["response"].get("body", {}).get("choices", [{}])[0].get("message", {}).get("content", "")

        
        parsed = extract_json(output_text)
        
        is_legal_rule = "Error"
        relevance_score = 0
        explanation = ""
        
        if parsed:
            is_legal_rule = parsed.get("is_legal_rule", "Error")
            relevance_score = parsed.get("relevance_score", 0)
            explanation = parsed.get("explanation", "")
        else:
            explanation = f"Failed to parse JSON. Raw output: {output_text[:100]}..."
            
        processed_data.append({
            "case_name": case_name,
            "claim": claim,
            "is_legal_rule": is_legal_rule,
            "relevance_score": relevance_score,
            "explanation": explanation,
            "raw_output": output_text
        })
            
    df = pd.DataFrame(processed_data)
    print(f"Processed {len(df)} results.")
    df.to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    
    gen_parser = subparsers.add_parser("generate")
    gen_parser.add_argument("--claims-file", default="../politifact_sc_matches_scraped.csv")
    gen_parser.add_argument("--metadata-file", default="../clean_data_with_details.csv")
    gen_parser.add_argument("--output-prompts", default="../legal_rule_prompts.jsonl")
    gen_parser.add_argument("--output-metadata", default="../legal_rule_metadata.jsonl")

    proc_parser = subparsers.add_parser("process")
    proc_parser.add_argument("--results-file", default="../legal_rule_output.jsonl")
    proc_parser.add_argument("--claims-file", default="../politifact_sc_matches_scraped.csv")
    proc_parser.add_argument("--output-csv", default="../results/legal_rule_results.csv")
    
    args = parser.parse_args()
    
    if args.mode == "generate":
        generate_prompts(args.claims_file, args.metadata_file, args.output_prompts, args.output_metadata)
    elif args.mode == "process":
        process_results(args.results_file, args.claims_file, args.output_csv)
