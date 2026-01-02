import pandas as pd
import json
from tqdm import tqdm
import argparse
import sys
import os
import re

# Prompt template for rewriting refuted claims
REWRITE_PROMPT_TEMPLATE = """# Instructions:
Rewrite the following legal claim to be a concise, general legal principle.
The input claim is a "refuted" (false) legal claim, but it is currently might be too long, specific, or conditional. The case the claim originated from is provided as context. It will contradict the claim, do not change the meaning of the claim.
Your task is to rewrite it so it is:
1. Independent of specific case details or parties (remove names, dates, specific locations).
2. Unconditional (remove "unless", "especially if", or specific factual caveats).
3. Concise and direct (simple, everyday language).
4. Focused on the core legal principle being asserted (even if that principle is false).

You must not change the meaning of the claim. If some details are necessary to preserve the meaning, keep them, even if that makes the claim lengthy.

If the claim is already concise and general, you may return it as is or with minor improvements.

## Input Claim:
"{claim}"

## Output Format:
Return a JSON object with a single key "rewritten_claim":
```json
{{
    "rewritten_claim": "..."
}}
```

## Case Evidence:
Facts: {facts}
Question: {question}
Conclusion: {conclusion}
"""

def create_openai_message(claim: str, facts: str, question: str, conclusion: str, custom_id: str = None) -> dict:
    """Create a single-user message in OpenAI chat JSON format"""
    formatted_prompt = REWRITE_PROMPT_TEMPLATE.format(
        claim=claim,
        facts=facts,
        question=question,
        conclusion=conclusion
    )
    msg = {"body": {"messages": [{"role": "user", "content": formatted_prompt}]}}
    if custom_id is not None:
        msg["custom_id"] = str(custom_id)
    return msg

def generate_rewrite_prompts(input_csv: str, metadata_csv: str, output_jsonl: str):
    """
    Reads the input CSV, filters for Refuted claims, joins with metadata, and generates prompts for rewriting.
    """
    print(f"Loading {input_csv}...")
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found.")
        return

    df = pd.read_csv(input_csv)
    # Preserve original index for mapping back results
    df['original_index'] = df.index
    
    print(f"Loading metadata from {metadata_csv}...")
    if not os.path.exists(metadata_csv):
        print(f"Error: {metadata_csv} not found.")
        return
    meta_df = pd.read_csv(metadata_csv)

    # Parse case_name to get a clean name for joining
    def get_clean_name(x):
        try:
            if isinstance(x, str) and x.startswith('['):
                val = json.loads(x)
                if isinstance(val, list) and len(val) > 0:
                    return val[0]
            return x
        except:
            return x

    df['clean_case_name'] = df['case_name'].apply(get_clean_name)
    
    # Merge with metadata
    merged_df = df.merge(meta_df[['name', 'facts', 'api_question', 'api_conclusion']], 
                         left_on='clean_case_name', right_on='name', how='left')

    # Filter for Refuted claims
    if 'label' in merged_df.columns:
        refuted_df = merged_df[merged_df['label'] == 0]
    elif 'is_negation' in merged_df.columns:
        refuted_df = merged_df[merged_df['is_negation'] == True]
    else:
        print("Error: Could not identify Refuted claims (missing 'label' or 'is_negation' column).")
        return

    print(f"Found {len(refuted_df)} refuted claims to process out of {len(df)} total.")

    entries = []
    
    for idx, row in tqdm(refuted_df.iterrows(), total=len(refuted_df), desc="Creating prompts"):
        claim = row['claim']
        if pd.isna(claim):
            continue
        
        facts = row.get('facts', 'N/A')
        question = row.get('api_question', 'N/A')
        conclusion = row.get('api_conclusion', 'N/A')
        original_idx = row.get('original_index', idx)
            
        msg = create_openai_message(claim, facts, question, conclusion, custom_id=original_idx)
        entries.append(msg)

    print(f"Writing {len(entries)} prompts to {output_jsonl}...")
    with open(output_jsonl, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')

    print(f"Created {len(entries)} prompts.")
    print("Run your vLLM or OpenAI batch inference using this file.")

def process_rewrite_results(jsonl_results_file: str, input_csv: str, output_csv: str):
    """
    Merges the rewritten claims back into the dataset.
    """
    print(f"Loading original data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Identify the rows we processed
    if 'label' in df.columns:
        mask = df['label'] == 0
    elif 'is_negation' in df.columns:
        mask = df['is_negation'] == True
    else:
        print("Error: Could not identify Refuted claims.")
        return

    refuted_indices = df[mask].index
    
    # Build a map of claim text to indices for fallback matching
    claim_to_indices = {}
    for idx in refuted_indices:
        c = df.at[idx, 'claim']
        if c not in claim_to_indices:
            claim_to_indices[c] = []
        claim_to_indices[c].append(idx)

    print(f"Loading results from {jsonl_results_file}...")
    results = []
    with open(jsonl_results_file, 'r') as f:
        for line in f:
            results.append(json.loads(line))
            
    # Create a new column for the rewritten claim
    df['rewritten_claim'] = df['claim']
    
    print("Updating claims...")
    
    matched_count = 0
    sequential_fallback_needed = False
    
    for i, res in enumerate(tqdm(results, desc="Processing results")):
        # Extract text from vLLM/OpenAI output format
        content = ""
        try:
            if 'response' in res and 'body' in res['response']:
                 content = res['response']['body']['choices'][0]['message']['content']
            elif 'outputs' in res:
                 content = res['outputs'][0]['text']
            elif 'answer' in res: # Handle the format we saw in head
                 content = res['answer']
            else:
                 content = str(res)
        except Exception as e:
            print(f"Error extracting content for result {i}: {e}")
            continue

        # Parse JSON from content
        rewritten_text = None
        try:
            json_match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
                rewritten_text = data.get("rewritten_claim")
            else:
                # Try parsing raw content if it's just json
                try:
                    data = json.loads(content)
                    if isinstance(data, dict):
                        rewritten_text = data.get("rewritten_claim")
                except:
                    pass
        except:
            pass
            
        if not rewritten_text:
            # Fallback: clean up quotes
            rewritten_text = content.strip().strip('"')

        # Determine target indices
        target_indices = []
        
        # 1. Try custom_id
        if 'custom_id' in res:
            try:
                target_indices = [int(res['custom_id'])]
            except:
                pass
        
        # 2. Try matching by prompt text (if custom_id missing)
        if not target_indices and 'prompt' in res:
            # Regex to extract the input claim from the prompt
            # Matches: ## Input Claim:\n"{claim}"
            match = re.search(r'## Input Claim:\s*\n"(.*?)"\s*\n\n## Output Format:', res['prompt'], re.DOTALL)
            if match:
                claim_text = match.group(1)
                if claim_text in claim_to_indices:
                    target_indices = claim_to_indices[claim_text]
        
        if target_indices:
            for idx in target_indices:
                if idx in df.index:
                    df.at[idx, 'rewritten_claim'] = rewritten_text
            matched_count += 1
        else:
            sequential_fallback_needed = True

    if matched_count == 0 and len(results) > 0:
        print("Warning: No results could be matched by ID or text. Falling back to sequential mapping.")
        print("CAUTION: This assumes results are in the exact same order as the filtered CSV rows.")
        
        limit = min(len(results), len(refuted_indices))
        for i in range(limit):
            res = results[i]
            original_idx = refuted_indices[i]
            
            # Extract content again (simplified)
            content = ""
            try:
                if 'response' in res and 'body' in res['response']:
                     content = res['response']['body']['choices'][0]['message']['content']
                elif 'outputs' in res:
                     content = res['outputs'][0]['text']
                elif 'answer' in res:
                     content = res['answer']
                else:
                     content = str(res)
            except:
                continue
                
            # Parse (simplified)
            rewritten_text = content.strip()
            try:
                json_match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group(1))
                    rewritten_text = data.get("rewritten_claim", rewritten_text)
            except:
                pass
                
            df.at[original_idx, 'rewritten_claim'] = rewritten_text
            
    elif matched_count < len(results):
        print(f"Matched {matched_count} results out of {len(results)}.")
        if sequential_fallback_needed:
             print("Some results could not be matched. They were skipped.")

    df.to_csv(output_csv, index=False)
    print(f"Saved updated dataset to {output_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rewrite Refuted claims to be concise and general.')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate prompts for rewriting')
    gen_parser.add_argument('--input-csv', default='../diff_case_claims_with_negatives_fixed.csv', help='Input dataset')
    gen_parser.add_argument('--metadata-csv', default='../clean_data_with_details.csv', help='Metadata dataset with facts/questions')
    gen_parser.add_argument('--output-jsonl', default='rewrite_prompts.jsonl', help='Output prompts file')

    # Process command
    proc_parser = subparsers.add_parser('process', help='Process results and update CSV')
    proc_parser.add_argument('--input-csv', default='../diff_case_claims_with_negatives_fixed.csv', help='Original input dataset')
    proc_parser.add_argument('--results-jsonl', default='../rewrite_results.jsonl', help='Results from LLM inference')
    proc_parser.add_argument('--output-csv', default='../diff_case_claims_rewritten.csv', help='Output CSV with rewritten claims')

    args = parser.parse_args()

    if args.command == 'generate':
        generate_rewrite_prompts(args.input_csv, args.metadata_csv, args.output_jsonl)
    elif args.command == 'process':
        process_rewrite_results(args.results_jsonl, args.input_csv, args.output_csv)
