import pandas as pd
import json
from tqdm import tqdm
import argparse
import subprocess
import sys
from typing import Optional, List, Dict
from openai import OpenAI

# prompt_template = """# Instructions:
# Carefully generate truthful factual claims for this court case using natural-sounding language while avoiding highly technical language.
# Make claims that extrapolate the implications of the court case, not specifically about the parties or specific facts of the case.
# Adhere to the following rules:
# - Do not use any part of the case name or other identifying information in the claim.
# - Don't make factual claims that are only applicable to the parties involved in the case. They have to be generalizable legal precedents.
# - Make the claims natural and easy to understand, like a real claim, without much legal jargon.
# - The claims should be generalizable.
# - The claims should have a reasonable length and only one central idea. Other claims can be included in subsequent claims.
# - The core idea of each claim should be significantly distinct from the others. Do not generate additional claims if all core ideas have already been covered in previous claims.
 
# ## Output Format:
# Return the claim in a JSON object with the following format:
# ```json
# {{
#     "claim1": "...",
#     "claim2": "...",
#     ...
# }}
# ```

# ## Facts:
# {facts}
 
# # Question:
# {question}

# # Conclusion:
# {conclusion}
# """

prompt_template = """# Instructions:
Generate truthful, factual claims about the legal implications of this case using simple, everyday language.
Avoid legal jargon or court-report phrasing. Do not use phrases like “the court found,” “the decision clarified,” or “under this test.”
Write claims as clear statements of what the law allows or does not allow.
Adhere to the following rules:
- Do not use any part of the case name or other identifying information in the claim.
- Do not make claims that only apply to the parties in this case; claims must be general legal principles.
- Keep each claim focused on one central idea and make it easy to understand.
- Each claim must be distinct from the others; do not repeat the same core idea.
- Use direct, plain wording rather than legal formulations.

## Output Format:
Return the claim in a JSON object with the following format:
```json
{{
    "claim1": "...",
    "claim2": "...",
    ...
}}
```

## Facts:
{facts}
 
# Question:
{question}

# Conclusion:
{conclusion}
"""

def create_openai_message(facts: str, question: str, conclusion: str) -> dict:
    """Create a single-user message in OpenAI chat JSON format"""
    formatted_prompt = prompt_template.format(
        facts=facts,
        question=question,
        conclusion=conclusion
    )
    return {"body": {"messages": [{"role": "user", "content": formatted_prompt}]}}


def prompt_vllm(prompt: str, model: str = 'Qwen/Qwen3-Next-80B-A3B-Thinking-FP8', base_url: str = 'http://localhost:8000/v1', think: bool = True) -> str:
    """Call a local vLLM OpenAI-compatible server and return text content.

    If `think` is False, injects a system message '/no_think' per provided pattern.
    """
    client = OpenAI(api_key='EMPTY', base_url=base_url)

    messages: List[Dict[str, str]] = []
    if not think:
        messages.append({'role': 'system', 'content': '/no_think'})
    messages.append({'role': 'user', 'content': prompt})

    try:
        response = client.chat.completions.create(model=model, messages=messages)
    except Exception as e:
        raise RuntimeError(f"Failed to reach vLLM server at {base_url}: {e}") from e

    output = response.choices[0].message.content
    if isinstance(output, str) and '<think>' in output[:20]:
        output = output.split('</think>')[-1]
    return output


def create_jsonl_dataset(input_csv: str, output_jsonl: str) -> int:
    """Generate a JSONL file; returns number of entries."""
    print(f"Loading {input_csv}...")
    df = pd.read_csv(input_csv).head(5)
    entries = []

    print(f"Processing {len(df)} cases...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating prompts"):
        if pd.isna(row.get('api_question')) or pd.isna(row.get('api_conclusion')) or pd.isna(row.get('facts')):
            continue

        msg = create_openai_message(
            facts=row['facts'],
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


def _extract_claims_from_text(content: str) -> Dict[str, Optional[str]]:
    """Parse up to 5 claims from model content; try JSON first, then heuristic fallback."""
    claims: Dict[str, Optional[str]] = {f"claim{i}": None for i in range(1, 6)}
    # Try strict JSON
    try:
        obj = json.loads(content)
        if isinstance(obj, dict):
            for i in range(1, 6):
                key = f"claim{i}"
                if key in obj and isinstance(obj[key], str):
                    claims[key] = obj[key]
            return claims
    except json.JSONDecodeError:
        pass

    # Heuristic: scan for "claimN": "..."
    for i in range(1, 6):
        pat = f'"claim{i}"'
        if pat in content:
            start = content.find(pat) + len(pat)
            start = content.find('"', start) + 1
            end = content.find('"', start)
            if end > start:
                claims[f"claim{i}"] = content[start:end]
    return claims


def generate_with_vllm(input_csv: str, model: str, max_rows: int = 5, output_csv: Optional[str] = None, base_url: str = 'http://localhost:8000/v1', think: bool = True) -> int:
    """Generate claims directly via vLLM OpenAI server for a small batch.

    - Reads the first `max_rows` rows from `input_csv` that have facts/question/conclusion.
    - Calls vLLM for each prompt.
    - Prints results and optionally saves to `output_csv` with columns generated_claim1..5 and raw_response.
    Returns number of processed rows.
    """
    print(f"Loading {input_csv}...")
    df_src = pd.read_csv(input_csv)
    df = df_src.head(max_rows).copy()

    # Initialize output columns
    df['claims_json'] = None
    df['raw_response'] = None

    valid_rows = 0
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating with vLLM"):
        if pd.isna(row.get('api_question')) or pd.isna(row.get('api_conclusion')) or pd.isna(row.get('facts')):
            continue

        formatted_prompt = prompt_template.format(
            facts=str(row['facts']),
            question=str(row['api_question']),
            conclusion=str(row['api_conclusion'])
        )

        try:
            content = prompt_vllm(prompt=formatted_prompt, model=model, base_url=base_url, think=think)
        except RuntimeError as e:
            print(f"[Error] Row {idx}: {e}")
            continue

        df.at[idx, 'raw_response'] = content
        claims = _extract_claims_from_text(content)
        claims_out = {f"Claim{i}": claims.get(f'claim{i}') for i in range(1, 6) if claims.get(f'claim{i}')}
        df.at[idx, 'claims_json'] = json.dumps(claims_out, ensure_ascii=False)

        # Print a quick summary to console
        print("\n==== Generated Claims (row {}) ====".format(idx))
        print(df.at[idx, 'claims_json'])
        valid_rows += 1

    # Always write CSV output
    output_path = output_csv or 'test.csv'
    df.to_csv(output_path, index=False)
    print(f"\nSaved results with generated claims to {output_path}.")

    print(f"Processed {valid_rows} rows via vLLM model '{model}'.")
    return valid_rows


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
        description='Generate a few test claims via local vLLM OpenAI server.'
    )
    parser.add_argument('--input-csv', default='clean_data_with_details.csv', help='Input CSV with case data')
    parser.add_argument('--model', default='Qwen/Qwen3-Next-80B-A3B-Thinking-FP8', help='Model name exposed by vLLM (e.g., Qwen/Qwen3-8B)')
    parser.add_argument('--max-rows', type=int, default=5, help='Max number of rows to process')
    parser.add_argument('--output-csv', default='test.csv', help='Path to save results CSV (default: test.csv)')
    parser.add_argument('--base-url', default='http://localhost:8000/v1', help='Base URL for the OpenAI-compatible server')
    parser.add_argument('--no-think', action='store_true', help='Disable thinking by sending /no_think system message')
    args = parser.parse_args()

    try:
        n = generate_with_vllm(
            input_csv=args.input_csv,
            model=args.model,
            max_rows=args.max_rows,
            output_csv=args.output_csv,
            base_url=args.base_url,
            think=(not args.no_think),
        )
    except RuntimeError as e:
        print(str(e))
        sys.exit(1)

    if n == 0:
        print("No rows processed (missing fields or errors). Exiting.")
        sys.exit(1)
    else:
        print("Done.")
