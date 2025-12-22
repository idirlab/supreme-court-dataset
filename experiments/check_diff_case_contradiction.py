import argparse
import pandas as pd
import json
import re
import os
from tqdm import tqdm

# Contradiction Prompt Template (from Cell 8)
PROMPT_TEMPLATE = """You are a legal expert. Read two short claims from different cases and the two cases' associated facts, legal questions, and conclusions.
Decide whether the two claims are contradicting (negation or opposite entailment), or consistent (when they are unrelated or entail).
You must output an explanation for your decision in the "explanation" field. Then, also provide a decision in the "contradiction" field: "contradiction" if the claims are contradicting, and "consistent" if they are not.

## Output Format:
Return a JSON object in the following format:
```json
{{
    "explanation": "...",
    "contradiction": "<contradiction/consistent>",
    ...
}}
```

Claim 1: {claim1}
Claim 2: {claim2}

Claim 1 Case Evidence:
Facts: {facts1}
Legal Question: {api_question1}
Conclusion: {api_conclusion1}

Claim 2 Case Evidence:
Facts: {facts2}
Legal Question: {api_question2}
Conclusion: {api_conclusion2}
"""

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

def generate_prompts(pairs_file, metadata_file, output_file, score_range=None):
    print(f"Loading pairs from {pairs_file}...")
    # Assuming pairs_file is a CSV with columns like claim1, docket1, name1, claim2, docket2, name2, score
    # Or at least docket1, name1, docket2, name2, claim1, claim2
    pairs_df = pd.read_csv(pairs_file)
    
    if score_range:
        print(f"Filtering pairs with score in range {score_range}...")
        pairs_df = pairs_df[(pairs_df['score'] >= score_range[0]) & (pairs_df['score'] <= score_range[1])]
        print(f"Filtered to {len(pairs_df)} pairs.")
    
    print(f"Loading metadata from {metadata_file}...")
    court_cases = pd.read_csv(metadata_file)
    
    print("Merging metadata...")
    # Merge for Claim 1
    # Similar to Cell 6
    df_with_facts = pairs_df.merge(
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
    # Drop the merge keys if they duplicate or interfere, but keep original identifiers
    # The notebook drops 'docket' and 'name' which come from the right table
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

    # Ensure we have the necessary columns for the prompt
    required_cols = ['claim1', 'claim2', 'facts1', 'api_question1', 'api_conclusion1', 'facts2', 'api_question2', 'api_conclusion2']
    # Fill NaNs to avoid format errors
    for col in required_cols:
        if col not in df_with_facts.columns:
            print(f"Warning: Column {col} missing.")
            df_with_facts[col] = ""
        df_with_facts[col] = df_with_facts[col].fillna("")

    print("Generating prompts...")
    df_with_facts['prompt'] = df_with_facts.apply(lambda row: PROMPT_TEMPLATE.format(**row), axis=1)
    
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
    
    print("Processing results...")
    
    # Check if lengths match
    if len(input_df) != len(output_df):
        print(f"Warning: Length mismatch. Input: {len(input_df)}, Output: {len(output_df)}")
        # We might need to merge on some ID if available, but usually these are line-aligned
        # If they are not line aligned, this is risky. Assuming line alignment as per notebook.
    
    # Merge relevant info from input to output if needed, or just use output if it carries over input fields
    # The notebook assumes line-by-line correspondence and copies columns
    
    # We'll create a combined dataframe
    # Start with input columns we want to keep
    combined = input_df.copy()
    
    # Add answer from output
    # Assuming output_df has 'answer' or 'outputs'
    if 'answer' in output_df.columns:
        combined['llm_output'] = output_df['answer']
    elif 'outputs' in output_df.columns:
        combined['llm_output'] = output_df['outputs'].apply(lambda x: x[0]['text'] if isinstance(x, list) and len(x) > 0 else "")
    elif 'choices' in output_df.columns:
        combined['llm_output'] = output_df['choices'].apply(lambda x: x[0]['message']['content'] if isinstance(x, list) and len(x) > 0 else "")
    else:
        print("Error: Could not find answer/outputs/choices in results file.")
        return

    # Parse JSON
    combined['parsed'] = combined['llm_output'].apply(extract_json)
    combined['decision'] = combined['parsed'].apply(lambda x: x.get('contradiction', 'error'))
    combined['explanation'] = combined['parsed'].apply(lambda x: x.get('explanation', ''))
    
    # Drop the temporary 'parsed' column if you don't want it in CSV, or keep it.
    # The user wants "similar to check_claims_overlap" which outputs a CSV.
    # We'll drop 'parsed' and 'llm_output' (maybe keep explanation) to keep it clean, 
    # or keep 'llm_output' for debugging. 
    # The notebook keeps: score, claim1, claim2, decision, explanation.
    # We should probably keep metadata too.
    
    cols_to_save = [c for c in combined.columns if c not in ['parsed', 'prompt']]
    final_df = combined[cols_to_save]
    
    print(f"Saving {len(final_df)} rows to {output_csv}...")
    final_df.to_csv(output_csv, index=False)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    
    gen_parser = subparsers.add_parser("generate")
    gen_parser.add_argument("--pairs-file", required=True, help="CSV file containing claim pairs (e.g., similarity_report.csv)")
    gen_parser.add_argument("--metadata-file", default="../clean_data_with_details.csv", help="CSV file with case metadata")
    gen_parser.add_argument("--output-file", required=True, help="Output JSONL file for LLM inference")
    gen_parser.add_argument("--range", nargs=2, type=float, default=[0.7032, 0.9491], help="Score range to filter pairs (min max)")

    proc_parser = subparsers.add_parser("process")
    proc_parser.add_argument("--input-file", required=True, help="The JSONL file generated in the generate step (used for metadata)")
    proc_parser.add_argument("--results-file", required=True, help="The JSONL file output from the LLM")
    proc_parser.add_argument("--output-csv", required=True, help="Output CSV file")
    
    args = parser.parse_args()

    if args.mode == "generate":
        generate_prompts(args.pairs_file, args.metadata_file, args.output_file, args.range)
    elif args.mode == "process":
        process_results(args.input_file, args.results_file, args.output_csv)