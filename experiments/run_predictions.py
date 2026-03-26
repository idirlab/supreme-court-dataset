import argparse
import json
import os
import sys
from openai import OpenAI
from tqdm import tqdm

def load_prompts(prompts_file):
    prompts = []
    with open(prompts_file, 'r') as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))
    return prompts

def extract_prompt_content(prompt_data):
    # Expected format: {"body": {"messages": [{"role": "user", "content": "..."}]}}
    try:
        return prompt_data['body']['messages'][0]['content']
    except (KeyError, IndexError, TypeError):
        # Fallback if format is different
        return str(prompt_data)

def run_predictions(prompts_file, output_file, mode, model, api_key=None, raw_output_file=None, sample_size=None):
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    client = OpenAI()
    
    prompts = load_prompts(prompts_file)
    print(f"Loaded {len(prompts)} prompts.")
    
    if sample_size:
        prompts = prompts[:sample_size]
        print(f"Running on a sample of {len(prompts)} prompts.")
    
    # Check if output file exists to resume? 
    # For now, we'll just append or overwrite. Let's overwrite to be safe/simple.
    # Or maybe append if we want to be robust.
    # Let's just open in 'w' mode.
    
    f_raw = None
    if raw_output_file:
        directory = os.path.dirname(raw_output_file)
        if directory:
            os.makedirs(directory, exist_ok=True)
        f_raw = open(raw_output_file, 'w')

    try:
        with open(output_file, 'w') as f:
            for i, prompt_data in enumerate(tqdm(prompts)):
                content = extract_prompt_content(prompt_data)
                
                try:
                    raw_record = None
                    if mode == "search":
                        # Use Responses API with web_search
                        response = client.responses.create(
                            model=model,
                            tools=[{"type": "web_search"}],
                            input=content
                        )
                        # Format for predict_and_evaluate.py
                        # It accepts {"response": "content string"}
                        output_record = {"response": response.output_text}
                        
                        # Try to get full dict
                        if hasattr(response, 'model_dump'):
                            raw_record = response.model_dump()
                        else:
                            # Fallback or just store what we can
                            raw_record = {"output_text": response.output_text}

                    else: # no-search
                        # Use Chat Completions API
                        response = client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": content}]
                        )
                        # Format for predict_and_evaluate.py
                        # It accepts standard chat completion JSON {"choices": [...]}
                        output_record = response.model_dump()
                        raw_record = output_record
                    
                    f.write(json.dumps(output_record) + "\n")
                    f.flush()
                    
                    if f_raw and raw_record:
                        f_raw.write(json.dumps(raw_record) + "\n")
                        f_raw.flush()
                    
                except Exception as e:
                    print(f"Error processing prompt {i}: {e}")
                    # Write an error record so line numbers match
                    f.write(json.dumps({"error": str(e), "verdict": "Error"}) + "\n")
                    f.flush()
                    if f_raw:
                        f_raw.write(json.dumps({"error": str(e)}) + "\n")
                        f_raw.flush()
    finally:
        if f_raw:
            f_raw.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run predictions using OpenAI API.")
    parser.add_argument("--prompts_file", default="./test_naive_llm_prompts.jsonl", help="Path to input prompts JSONL file.")
    parser.add_argument("--output_file", required=True, help="Path to output predictions JSONL file.")
    parser.add_argument("--mode", choices=["search", "no-search"], required=True, help="Mode: search (Responses API) or no-search (Chat Completions).")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model to use (default: gpt-4o).")
    parser.add_argument("--api_key", help="OpenAI API Key (optional, can use env var).")
    parser.add_argument("--raw_output_file", default= "../data_store/raw_results/openai_no_search_raw_output_2.jsonl", help="Path to raw output JSONL file (optional).")
    parser.add_argument("--sample_size", type=int, help="Number of prompts to run (optional).")
    
    args = parser.parse_args()
    
    run_predictions(args.prompts_file, args.output_file, args.mode, args.model, args.api_key, args.raw_output_file, args.sample_size)
