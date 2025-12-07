import json
import pandas as pd
import re
from concurrent.futures import ThreadPoolExecutor
import argparse
from typing import Any, Dict, List

JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
# Fallback: any JSON object containing at least claim1 key; we will parse whatever claims exist
FALLBACK_JSON_RE = re.compile(r"(\{[^{}]*\"claim1\"[\s\S]*?\})", re.DOTALL)


def normalize_response(output: Any) -> str:
    """Normalize various output formats to a string.
    Accepts:
      - plain string
      - dict with key 'output' being string
      - vLLM-like dict: {'outputs': [{'text': '...'}], ...}
      - any None or unexpected type -> empty string
    """
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        # Common cases
        if "outputs" in output and isinstance(output["outputs"], list) and output["outputs"]:
            text = output["outputs"][0].get("text")
            if isinstance(text, str):
                return text
        if "output" in output and isinstance(output["output"], str):
            return output["output"]
        # Fallback: stringify
        try:
            return json.dumps(output, ensure_ascii=False)
        except Exception:
            return str(output)
    # Fallback for other types
    return str(output)


def _find_last_json_object(text: str) -> str | None:
    """Find the last JSON object in the text using brace balance.
    Returns the substring or None.
    """
    # Work from the end: locate last '{' and matching '}' by balance
    last_open = text.rfind('{')
    if last_open == -1:
        return None
    # From last_open to end, find matching close by counting braces
    balance = 0
    for i, ch in enumerate(text[last_open:], start=last_open):
        if ch == '{':
            balance += 1
        elif ch == '}':
            balance -= 1
            if balance == 0:
                return text[last_open:i+1]
    return None


def extract_json_claims(raw_response: str) -> Dict[str, str]:
    # Prefer explicit ```json block
    match = JSON_BLOCK_RE.search(raw_response)
    candidates: list[str] = []
    if match:
        candidates.append(match.group(1))

    # Fallback regex for objects containing claim1
    curly_match = FALLBACK_JSON_RE.search(raw_response)
    if curly_match:
        candidates.append(curly_match.group(1))

    # Heuristic: take the last JSON-looking object in the text
    last_obj = _find_last_json_object(raw_response)
    if last_obj:
        candidates.append(last_obj)

    # Try candidates in reverse order (most recent appearance first)
    for json_str in reversed(candidates):
        if not json_str:
            continue
        try:
            clean = json_str.replace('""', '"').strip()
            parsed = json.loads(clean)
            # Collect up to 10 claims if available
            result: Dict[str, str] = {}
            for i in range(1, 11):
                key = f"claim{i}"
                if key in parsed and isinstance(parsed[key], str):
                    result[f"claim_{i}"] = parsed[key].strip()
            # Ensure at least claim_1..claim_5 keys exist
            for i in range(1, 6):
                result.setdefault(f"claim_{i}", "")
            return result
        except json.JSONDecodeError:
            continue

    # No JSON parsed
    return {f"claim_{i}": "" for i in range(1, 6)}


def process_line(item: Dict[str, Any]) -> Dict[str, Any]:
    raw_output = item.get("output")
    raw_response = normalize_response(raw_output)
    claims = extract_json_claims(raw_response)
    return {
        "raw_response": raw_response.strip(),
        **claims
    }


def process_batch_results(jsonl_path: str, output_csv_path: str, metadata_csv_path: str | None = None):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]

    with ThreadPoolExecutor() as executor:
        results: List[Dict[str, Any]] = list(executor.map(process_line, lines))

    # DataFrame with raw responses and extracted claims (variable number of claim_* keys)
    df = pd.DataFrame(results)

    # Count responses with no non-empty claims
    claim_cols = [c for c in df.columns if c.startswith("claim_")]
    num_no_claims = int((df[claim_cols].replace("", pd.NA).dropna(how="all", axis=1)  # keep any non-empty claim cols
                        .reindex(columns=claim_cols)
                        .isna()).all(axis=1).sum()) if claim_cols else len(df)
    print(f"Raw responses without claims: {num_no_claims} / {len(df)}")

    # 1) Save raw responses CSV to ./data/datasets/<input_basename>.csv
    import os
    base = os.path.splitext(os.path.basename(jsonl_path))[0]
    raw_out_dir = os.path.join(".", "data", "datasets")
    os.makedirs(raw_out_dir, exist_ok=True)
    raw_out_path = os.path.join(raw_out_dir, f"{base}.csv")
    pd.DataFrame({"raw_response": df["raw_response"]}).to_csv(raw_out_path, index=False)
    print(f"Saved raw responses to: {raw_out_path}")

    # 2) Save claims CSV in current directory named by input (base)_claims.csv
    claims_out_path = os.path.join(".", f"{base}_claims.csv")

    claims_df = df[["raw_response"] + claim_cols].copy()

    # Merge selected metadata columns if provided and present
    if metadata_csv_path:
        try:
            metadata_df = pd.read_csv(metadata_csv_path)
            # Try to include common identifiers
            meta_cols = [c for c in metadata_df.columns if c.lower() in {"case_name", "docket", "court", "id"}]
            if not meta_cols:
                # If not found, include all metadata to preserve association
                meta_cols = list(metadata_df.columns)
            claims_df = pd.concat([metadata_df[meta_cols].reset_index(drop=True),
                                   claims_df.reset_index(drop=True)], axis=1)
        except Exception as e:
            print(f"Warning: Failed to merge metadata: {e}")

    claims_df.to_csv(claims_out_path, index=False)
    print(f"Saved claims to: {claims_out_path}")


def main():
    parser = argparse.ArgumentParser(description="Safely process batch vLLM results into CSV")
    parser.add_argument("--input", type=str, default="results.jsonl", help="Path to JSONL results")
    parser.add_argument("--output", type=str, default="sc-claims_v1.csv", help="Path to output CSV")
    parser.add_argument("--metadata", type=str, default="clean_data_with_details.csv", help="Optional metadata CSV to merge")
    args = parser.parse_args()

    process_batch_results(args.input, args.output, args.metadata)


if __name__ == "__main__":
    main()
