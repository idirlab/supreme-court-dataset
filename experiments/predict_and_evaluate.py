import argparse
import pandas as pd
import json
import os
import sys
import re

# Add parent directory to path to import eval_script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import eval_script
except ImportError:
    # Fallback if running from root or elsewhere
    sys.path.append(os.path.join(os.getcwd(), 'supreme-court-dataset'))
    try:
        import eval_script
    except ImportError:
        print("Warning: Could not import eval_script. Evaluation might fail.")

PROMPT_TEMPLATE = """You are a legal expert. Your task is to analyze a legal claim and determine its veracity based on US Supreme Court cases.
You must determine if the claim is "Supported", "Refuted", or "Overruled" by the case law.
You must also identify the specific Supreme Court cases that serve as evidence for your decision. List them in order of importance (most important first). You will be penalized for irrelevant and incorrect citations, so prioritize accuracy and conciseness of citations.

Constraints:
1. You must ONLY cite cases from the provided list of valid Supreme Court cases. Do not invent cases or cite cases not in the list.
2. Do not guess. If you are unsure, provide your best estimate but prioritize accuracy.
3. Output must be a valid JSON object.

Valid Supreme Court Cases:
{case_list}

Claim: {claim}

Respond with a JSON object in the following format:
{{
    "explanation": "Brief explanation of your reasoning.",
    "cases": ["Case Name 1", "Case Name 2", ...],
    "verdict": "Supported" or "Refuted" or "Overruled"
}}
"""

def create_openai_message(prompt):
    return {"body": {"messages": [{"role": "user", "content": prompt}]}}

def generate_prompts(test_set_path, cases_path, output_prompts, output_metadata):
    print(f"Loading test set from {test_set_path}...")
    test_df = pd.read_csv(test_set_path)
    
    print(f"Loading cases from {cases_path}...")
    cases_df = pd.read_csv(cases_path)
    
    # Get list of valid case names
    if 'name' not in cases_df.columns:
        print("Error: 'name' column not found in cases file.")
        return

    valid_cases = cases_df['name'].dropna().unique().tolist()
    # Sort for consistency
    valid_cases.sort()
    valid_cases_str = ", ".join(valid_cases)
    
    print(f"Found {len(valid_cases)} valid cases.")
    
    prompts = []
    metadata_records = []
    
    print("Generating prompts...")
    for idx, row in test_df.iterrows():
        claim = row['claim']
        
        prompt_text = PROMPT_TEMPLATE.format(
            case_list=valid_cases_str,
            claim=claim
        )
        
        prompts.append(create_openai_message(prompt_text))
        metadata_records.append({
            "claim_id": idx,
            "claim": claim,
            "gold_label": row.get('label'),
            "gold_class": row.get('class'),
            "gold_cases": row.get('case_name')
        })
        
    print(f"Writing {len(prompts)} prompts to {output_prompts}...")
    with open(output_prompts, 'w') as f:
        for p in prompts:
            f.write(json.dumps(p) + '\n')
            
    print(f"Writing metadata to {output_metadata}...")
    with open(output_metadata, 'w') as f:
        for m in metadata_records:
            f.write(json.dumps(m) + '\n')
    
    print("Done generating prompts.")

def extract_json_from_text(text):
    """Extract JSON object from text, handling markdown code blocks."""
    try:
        # Try to find JSON block
        match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        
        match = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
            
        # Try to find start and end of JSON object
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            return json.loads(text[start:end+1])
            
        return json.loads(text)
    except Exception:
        return None

def evaluate_predictions(predictions_file, metadata_file, test_set_path, output_csv):
    print(f"Loading predictions from {predictions_file}...")
    with open(predictions_file, 'r') as f:
        preds_lines = f.readlines()
        
    print(f"Loading metadata from {metadata_file}...")
    with open(metadata_file, 'r') as f:
        meta_lines = f.readlines()
        
    if len(preds_lines) != len(meta_lines):
        print(f"Warning: Number of predictions ({len(preds_lines)}) does not match metadata ({len(meta_lines)}).")
        # We will proceed with zip, which truncates to shortest
        
    results = []
    
    print("Parsing predictions...")
    for i, (pred_line, meta_line) in enumerate(zip(preds_lines, meta_lines)):
        meta = json.loads(meta_line)
        try:
            pred_data = json.loads(pred_line)
            
            # Determine where the content is
            content = None
            if isinstance(pred_data, dict):
                if 'response' in pred_data and isinstance(pred_data['response'], dict) and 'body' in pred_data['response']:
                     # Format: {"response": {"body": {"choices": [...]}}}
                     choices = pred_data['response']['body'].get('choices', [])
                     if choices:
                         content = choices[0]['message']['content']
                elif 'choices' in pred_data:
                    content = pred_data['choices'][0]['message']['content']
                elif 'response' in pred_data:
                    content = pred_data['response']
                elif 'verdict' in pred_data: # Direct JSON object
                    content = json.dumps(pred_data)
                else:
                    # Fallback: maybe the whole line is the content if it's a string?
                    # But json.loads(pred_line) returned a dict.
                    pass
            
            if content is None:
                # Maybe the line itself is the content string (if it was double encoded?)
                # Or maybe the user provided a file with just the text responses.
                # Let's assume the user provides a standard format or we try to parse the line as text.
                content = pred_line

            if isinstance(content, dict):
                parsed = content
            else:
                parsed = extract_json_from_text(str(content))
            
            if not parsed:
                raise ValueError("Could not extract JSON from response")
                
            verdict = parsed.get('verdict', 'Unknown')
            cases = parsed.get('cases', [])
            
            # Normalize verdict to match test_set classes if possible, 
            # but eval_script handles string comparison.
            
            results.append({
                "claim": meta['claim'],
                "predicted_verdict": verdict,
                "predicted_cases": cases
            })
            
        except Exception as e:
            print(f"Error parsing prediction for claim {i}: {e}")
            results.append({
                "claim": meta['claim'],
                "predicted_verdict": "Error",
                "predicted_cases": []
            })

    # Convert to DataFrame
    pred_df = pd.DataFrame(results)
    
    # Save to CSV
    pred_df.to_csv(output_csv, index=False)
    print(f"Saved parsed predictions to {output_csv}")
    
    # Run evaluation
    print("Running evaluation...")
    
    # We need to pass absolute paths to eval_script if we are not in the same dir
    # But eval_script uses pd.read_csv so paths are fine.
    
    # eval_script.evaluate_files returns a dict, we can print the summary
    eval_result = eval_script.evaluate_files(
        gold_path=test_set_path,
        pred_path=output_csv,
        pred_cases_col="predicted_cases",
        pred_label_col="predicted_verdict",
        output_path=output_csv.replace(".csv", "_eval_results.csv")
    )
    
    print("\nEvaluation Summary:")
    summary = eval_result['summary']
    for k, v in summary.items():
        if k != "per_sample_output":
            print(f"{k}: {v}")
    print(f"\nDetailed results saved to {summary['per_sample_output']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate prompts for claim verification and evaluate results.")
    parser.add_argument("--mode", choices=["generate", "evaluate"], required=True, help="Mode: generate prompts or evaluate predictions")
    parser.add_argument("--test_set", default="../test_set.csv", help="Path to test_set.csv")
    parser.add_argument("--cases", default="../supreme_court_cases.csv", help="Path to supreme_court_cases.csv")
    parser.add_argument("--prompts_output", default="../test_naive_llm_prompts.jsonl", help="Output path for prompts (generate mode)")
    parser.add_argument("--metadata_output", default="../test_naive_llm_metadata.jsonl", help="Output path for metadata (generate mode)")
    parser.add_argument("--predictions_file", help="Path to LLM output file (evaluate mode)")
    parser.add_argument("--results_csv", default="predictions_parsed.csv", help="Intermediate CSV for predictions (evaluate mode)")
    
    args = parser.parse_args()
    
    # Resolve paths relative to script location if they are relative
    # But user might run from anywhere. Let's trust the user paths or defaults.
    # If defaults are used, they assume running from experiments/ folder.
    
    if args.mode == "generate":
        generate_prompts(args.test_set, args.cases, args.prompts_output, args.metadata_output)
    elif args.mode == "evaluate":
        if not args.predictions_file:
            print("Error: --predictions_file is required for evaluate mode")
            sys.exit(1)
        evaluate_predictions(args.predictions_file, args.metadata_output, args.test_set, args.results_csv)
