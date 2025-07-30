import pandas as pd
import json
from tqdm import tqdm
import argparse
import subprocess
import sys

prompt_template = """# Instructions:
Carefully generate 5 truthful factual claims for this court case using natural-sounding language while avoiding highly technical language.
Make claims that extrapolate the implications of the court case, not specifically about the parties involved in the case.
Adhere to the following rules:
- Do not use any part of the case name in the claim.
- Don't make factual claims that are only applicable to the parties involved in the case.
- Make the claim natural and easy to understand, so have a reasonable length.
 
## Output Format:
Return the claim in a JSON object with the following format:
```json
{{
    "claim1": "...",
    "claim2": "...",
    "claim3": "...",
    "claim4": "...",
    "claim5": "..."
}}
```
 
## Facts:
{example_sentences}
 
# Question:
{question}

# Conclusion:
{conclusion}
"""

def create_openai_message(example_sentences: str, question: str, conclusion: str) -> dict:
    """Create a single-user message in OpenAI chat JSON format"""
    formatted_prompt = prompt_template.format(
        example_sentences=example_sentences,
        question=question,
        conclusion=conclusion
    )
    return {"body": {"messages": [{"role": "user", "content": formatted_prompt}]}}


def create_jsonl_dataset(input_csv: str, output_jsonl: str) -> int:
    """Generate a JSONL file; returns number of entries."""
    print(f"Loading {input_csv}...")
    df = pd.read_csv(input_csv)
    entries = []

    print(f"Processing {len(df)} cases...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating prompts"):
        if pd.isna(row.get('api_question')) or pd.isna(row.get('api_conclusion')) or pd.isna(row.get('facts')):
            continue

        msg = create_openai_message(
            example_sentences=row['facts'],
            question=row['api_question'],
            conclusion=row['api_conclusion']
        )
        entries.append(msg)

    print(f"Writing {len(entries)} prompts to {output_jsonl}...")
    with open(output_jsonl, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')

    print(f"Created {len(entries)} prompts.")
    return len(entries)


def process_batch_results(jsonl_results_file: str, output_csv: str, input_csv: str):
    """Process vLLM batch inference JSONL into CSV with claims and raw response."""
    print(f"Processing batch results from {jsonl_results_file}...")
    df = pd.read_csv(input_csv)

    # Initialize columns
    for i in range(1, 6):
        df[f'generated_claim{i}'] = None
    df['raw_response'] = None

    results = []
    with open(jsonl_results_file, 'r') as f:
        for line in f:
            results.append(json.loads(line))

    processed = 0
    for res in results:
        content = None
        # vLLM generate output format
        try:
            content = res['outputs'][0]['text']
        except Exception:
            print(f"Unable to extract text for entry: {res}")
            continue

        idx = processed  # assumes same order
        df.at[idx, 'raw_response'] = content
        # parse JSON claims
        try:
            claim_data = json.loads(content)
            for i in range(1, 6):
                key = f'claim{i}'
                if key in claim_data:
                    df.at[idx, f'generated_claim{i}'] = claim_data[key]
        except json.JSONDecodeError:
            # fallback simple extraction
            for i in range(1,6):
                pat = f'"claim{i}"'
                if pat in content:
                    start = content.find(pat) + len(pat)
                    start = content.find('"', start) + 1
                    end = content.find('"', start)
                    if end > start:
                        df.at[idx, f'generated_claim{i}'] = content[start:end]
        processed += 1

    df.to_csv(output_csv, index=False)
    print(f"Saved processed claims to {output_csv}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create prompts JSONL file for vLLM batch inference'
    )
    parser.add_argument('--input-csv', default='clean_data_with_details.csv', help='Input CSV with case data')
    parser.add_argument('--prompts-jsonl', default='prompts.jsonl', help='Output prompts JSONL file')
    args = parser.parse_args()

    # 1. Create prompts
    n = create_jsonl_dataset(args.input_csv, args.prompts_jsonl)
    if n == 0:
        print("No prompts to process. Exiting.")
        sys.exit(1)
    
    print(f"Successfully created {args.prompts_jsonl} with {n} prompts.")
    print("You can now run vLLM batch inference manually with this file.")
