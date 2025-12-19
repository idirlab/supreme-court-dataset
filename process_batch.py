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


def extract_prompt_text(item: Dict[str, Any]) -> str:
    # Try 'prompt' field
    if "prompt" in item and isinstance(item["prompt"], str):
        return item["prompt"]
    
    # Try 'body' -> 'messages' (OpenAI chat format)
    if "body" in item and isinstance(item["body"], dict):
        body = item["body"]
        if "messages" in body and isinstance(body["messages"], list):
            for msg in body["messages"]:
                if msg.get("role") == "user":
                    return msg.get("content", "")
    
    # Try direct 'messages'
    if "messages" in item and isinstance(item["messages"], list):
        for msg in item["messages"]:
            if msg.get("role") == "user":
                return msg.get("content", "")
                
    return ""


def extract_conclusion(text: str) -> str:
    # Look for "# Conclusion:" followed by text
    if not text:
        return ""
    # Use non-greedy match to capture content before <|im_end|> or end of string
    match = re.search(r"# Conclusion:\s*(.*?)(?:<\|im_end\|>|$)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def process_line(item: Dict[str, Any]) -> Dict[str, Any]:
    raw_output = item.get("output")
    raw_response = normalize_response(raw_output)
    claims = extract_json_claims(raw_response)
    
    # Extract conclusion for matching
    prompt_text = extract_prompt_text(item)
    conclusion = extract_conclusion(prompt_text)

    return {
        "raw_response": raw_response.strip(),
        "extracted_conclusion": conclusion,
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

    # Long format: each row is a single claim with associated case_name and docket
    long_rows = []
    # Load metadata (optional) to attach case_name and docket
    metadata_df = None
    meta_case_col = None
    meta_docket_col = None
    conclusion_map = {}

    if metadata_csv_path:
        try:
            metadata_df = pd.read_csv(metadata_csv_path)
            # Normalize column names for matching
            lower_map = {c.lower(): c for c in metadata_df.columns}
            # Use 'name' column from metadata (case name)
            meta_case_col = lower_map.get("name")
            meta_docket_col = lower_map.get("docket")
            
            # Build conclusion map
            meta_conc_col = lower_map.get("api_conclusion")
            if meta_conc_col:
                for idx, m_row in metadata_df.iterrows():
                    val = m_row.get(meta_conc_col)
                    if isinstance(val, str) and val.strip():
                        conclusion_map[val.strip()] = m_row
            else:
                print("Warning: 'api_conclusion' column not found in metadata. Matching by conclusion disabled.")

        except Exception as e:
            print(f"Warning: Failed to load metadata: {e}")
            metadata_df = None

    # Iterate rows and collect non-empty claims
    for i, row in df.iterrows():
        case_name = None
        docket = None
        
        # Attempt match by conclusion
        extracted_conc = row.get("extracted_conclusion")
        if isinstance(extracted_conc, str) and extracted_conc and extracted_conc in conclusion_map:
            meta_row = conclusion_map[extracted_conc]
            if meta_case_col:
                case_name = meta_row.get(meta_case_col)
            if meta_docket_col:
                docket = meta_row.get(meta_docket_col)
        
        # Fallbacks
        case_name = case_name if isinstance(case_name, str) else ""
        docket = docket if isinstance(docket, str) else ""

        for c in claim_cols:
            val = row.get(c, "")
            if isinstance(val, float) and pd.isna(val):
                val = ""
            if isinstance(val, str):
                text = val.strip()
            else:
                text = str(val).strip() if val is not None else ""
            if text:
                long_rows.append({
                    "name": case_name,
                    "docket": docket,
                    "claim": text
                })

    claims_long_df = pd.DataFrame(long_rows, columns=["name", "docket", "claim"])
    claims_long_df.to_csv(claims_out_path, index=False)
    print(f"Saved claims (long format) to: {claims_out_path}")


def main():
    parser = argparse.ArgumentParser(description="Safely process batch vLLM results into CSV")
    parser.add_argument("--input", type=str, default="results_80B.jsonl", help="Path to JSONL results")
    parser.add_argument("--output", type=str, default="claims_raw.csv", help="Path to output CSV")
    parser.add_argument("--metadata", type=str, default="clean_data_with_details.csv", help="Optional metadata CSV to merge")
    args = parser.parse_args()

    process_batch_results(args.input, args.output, args.metadata)


if __name__ == "__main__":
    main()
