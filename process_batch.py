import json
import pandas as pd
import re
from concurrent.futures import ThreadPoolExecutor
import argparse

def extract_json_claims(raw_response):
    # Try to extract between triple backtick JSON block
    match = re.search(r"```json\s*(\{.*?\})\s*```", raw_response, re.DOTALL)
    json_str = None

    if match:
        json_str = match.group(1)
    else:
        # Fallback: first JSON-like block with at least 5 claims
        curly_match = re.search(r"(\{[^{}]*\"claim1\".*?\"claim5\"[^{}]*\})", raw_response, re.DOTALL)
        if curly_match:
            json_str = curly_match.group(1)

    if json_str:
        try:
            # Unescape double quotes if needed
            json_str = json_str.replace('""', '"')
            parsed = json.loads(json_str)
            return {f"claim_{i}": parsed.get(f"claim{i}", "").strip() for i in range(1, 6)}
        except json.JSONDecodeError:
            return {f"claim_{i}": "" for i in range(1, 6)}
    else:
        return {f"claim_{i}": "" for i in range(1, 6)}

def process_line(item):
    raw_response = item["output"]
    claims = extract_json_claims(raw_response)
    return {
        "raw_response": raw_response.strip(),
        **claims
    }

def process_batch_results(jsonl_path, output_csv_path, metadata_csv_path=None):
    with open(jsonl_path, "r") as f:
        lines = [json.loads(line) for line in f]

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_line, lines))
    
    df = pd.DataFrame(results)

    if metadata_csv_path:
        try:
            metadata_df = pd.read_csv(metadata_csv_path)
            df = pd.concat([metadata_df.reset_index(drop=True), df.reset_index(drop=True)], axis=1)
        except Exception as e:
            print(f"Warning: Failed to merge metadata: {e}")

    df.to_csv(output_csv_path, index=False)
    print(f"Saved {len(df)} results to: {output_csv_path}")


process_batch_results("outputs.jsonl", "sc-claims_v3.csv", "clean_data_with_details.csv")