import argparse
import pandas as pd
import json
import re
import os
import hashlib
from tqdm import tqdm

# Pairwise Resolution Prompt
PROMPT_TEMPLATE = """You are a legal expert. You are given two claims from different cases that have been identified as redundant (overlapping).
Your task is to decide how to resolve this overlap. You can:
1. Keep Claim 1 (if it is more accurate, comprehensive, or better phrased). The ground truth for Claim 1 will be both cases.
2. Keep Claim 2 (if it is more accurate, comprehensive, or better phrased). The ground truth for Claim 2 will be both cases.
3. Merge them (create a new claim that combines the information from both).

Claim 1: {claim1}
Claim 2: {claim2}

Case 1 Evidence:
Facts: {facts1}
Question: {api_question1}
Conclusion: {api_conclusion1}

Case 2 Evidence:
Facts: {facts2}
Question: {api_question2}
Conclusion: {api_conclusion2}

Output JSON:
{{
    "reasoning": "...",
    "decision": "keep_1" | "keep_2" | "merge",
    "merged_claim": "..." (only if decision is merge, otherwise null)
}}
"""

def create_openai_message(prompt):
    return {"body": {"messages": [{"role": "user", "content": prompt}]}}

def extract_json(text):
    try:
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        return {}
    except:
        return {}

def generate_prompts(input_csv, metadata_file, output_file):
    print(f"Loading overlapping pairs from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Filter for redundant pairs
    if 'decision' in df.columns:
        df = df[df['decision'] == 'redundant']
    
    # Filter out same case pairs
    if 'name1' in df.columns and 'name2' in df.columns:
        original_len = len(df)
        df = df[df['name1'] != df['name2']]
        print(f"Filtered out {original_len - len(df)} pairs from the same case.")

    print(f"Found {len(df)} redundant pairs.")
    
    # Drop existing metadata columns to avoid duplicates after merge
    cols_to_drop = ['facts1', 'api_question1', 'api_conclusion1', 
                    'facts2', 'api_question2', 'api_conclusion2']
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    
    print(f"Loading metadata from {metadata_file}...")
    court_cases = pd.read_csv(metadata_file)
    
    print("Merging metadata...")
    # Merge for Claim 1
    df_with_facts = df.merge(
        court_cases[['docket', 'name', 'facts', 'api_question', 'api_conclusion']],
        left_on=['docket1', 'name1'],
        right_on=['docket', 'name'],
        how='left'
    )
    
    df_with_facts.rename(
        columns={
            'facts': 'facts1',
            'api_question': 'api_question1',
            'api_conclusion': 'api_conclusion1'
        },
        inplace=True
    )
    df_with_facts.drop(columns=['docket', 'name'], inplace=True)

    # Merge for Claim 2
    df_with_facts = df_with_facts.merge(
        court_cases[['docket', 'name', 'facts', 'api_question', 'api_conclusion']],
        left_on=['docket2', 'name2'],
        right_on=['docket', 'name'],
        how='left'
    )

    df_with_facts.rename(
        columns={
            'facts': 'facts2',
            'api_question': 'api_question2',
            'api_conclusion': 'api_conclusion2'
        },
        inplace=True
    )
    df_with_facts.drop(columns=['docket', 'name'], inplace=True)

    # Fill NaNs
    required_cols = ['claim1', 'claim2', 'facts1', 'api_question1', 'api_conclusion1', 'facts2', 'api_question2', 'api_conclusion2']
    for col in required_cols:
        df_with_facts[col] = df_with_facts[col].fillna("")

    print("Generating prompts...")
    df_with_facts['prompt'] = df_with_facts.apply(lambda row: PROMPT_TEMPLATE.format(**row), axis=1)
    df_with_facts['body'] = df_with_facts['prompt'].apply(lambda x: create_openai_message(x)['body'])
    
    print(f"Writing {len(df_with_facts)} records to {output_file}...")
    df_with_facts.to_json(output_file, orient='records', lines=True)
    print("Done.")

def process_results(input_file, results_file, output_csv):
    print(f"Loading input data from {input_file}...")
    input_df = pd.read_json(input_file, lines=True)
    
    print(f"Loading results from {results_file}...")
    if not os.path.exists(results_file):
        print(f"Error: Results file {results_file} not found.")
        return

    output_df = pd.read_json(results_file, lines=True)
    
    if len(input_df) != len(output_df):
        print(f"Warning: Length mismatch. Input: {len(input_df)}, Output: {len(output_df)}")
    
    # Combine input and output
    combined = input_df.copy()
    if 'answer' in output_df.columns:
        combined['llm_output'] = output_df['answer']
    elif 'outputs' in output_df.columns:
        combined['llm_output'] = output_df['outputs'].apply(lambda x: x[0]['text'] if isinstance(x, list) and len(x) > 0 else "")
    elif 'choices' in output_df.columns:
        combined['llm_output'] = output_df['choices'].apply(lambda x: x[0]['message']['content'] if isinstance(x, list) and len(x) > 0 else "")
    
    combined['parsed'] = combined['llm_output'].apply(extract_json)
    combined['resolution_decision'] = combined['parsed'].apply(lambda x: x.get('decision', 'error'))
    combined['merged_claim'] = combined['parsed'].apply(lambda x: x.get('merged_claim', ''))
    combined['reasoning'] = combined['parsed'].apply(lambda x: x.get('reasoning', ''))
    
    cols_to_save = [c for c in combined.columns if c not in ['parsed', 'prompt', 'body']]
    
    print(f"Saving {len(combined)} rows to {output_csv}...")
    combined[cols_to_save].to_csv(output_csv, index=False)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    
    gen_parser = subparsers.add_parser("generate")
    gen_parser.add_argument("--input-csv", default="../results/overlap_diff_case.csv")
    gen_parser.add_argument("--metadata-file", default="../clean_data_with_details.csv")
    gen_parser.add_argument("--output-file", default="../resolve_overlap_prompts.jsonl")

    proc_parser = subparsers.add_parser("process")
    proc_parser.add_argument("--input-file", default="../resolve_overlap_prompts.jsonl")
    proc_parser.add_argument("--results-file", required=True)
    proc_parser.add_argument("--output-csv", default="../results/factual_claims_resolved_overlaps.csv")
    
    args = parser.parse_args()
    
    if args.mode == "generate":
        generate_prompts(args.input_csv, args.metadata_file, args.output_file)
    elif args.mode == "process":
        process_results(args.input_file, args.results_file, args.output_csv)
