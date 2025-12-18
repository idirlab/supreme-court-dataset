import argparse
import json
import os
import sys
import pandas as pd
from transformers import AutoTokenizer
from dotenv import load_dotenv

load_dotenv()

# Workaround for tvm_ffi PermissionError
for path in sys.path:
    if "site-packages" in path and os.path.isdir(os.path.join(path, "tvm_ffi")):
        os.environ["TVM_LIBRARY_PATH"] = os.path.join(path, "tvm_ffi")
        break

from vllm import LLM, SamplingParams

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google/gemma-3-27b-it")
    parser.add_argument("--prompts", type=str, default="contradiction_prompts.jsonl")
    parser.add_argument("--output", type=str, default="contradiction_output_vllm.jsonl")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size (number of GPUs to use)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading model {args.model} with TP={args.tp}...")
    
    # Load tokenizer for chat template
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Load prompts
    print(f"Reading prompts from {args.prompts}...")
    df = pd.read_json(args.prompts, lines=True)
    prompts = []
    for _, row in df.iterrows():
        prompt = tokenizer.apply_chat_template(
            row['body']["messages"],
            tokenize=False,
            add_generation_prompt=True,
            token=os.environ.get("HUGGINGFACE_TOKEN"),
        )
        prompts.append(prompt)

    # Initialize vLLM
    # 27B model fits on one H100 (80GB). 
    # Increasing tp (tensor_parallel_size) will split the model across GPUs.
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        trust_remote_code=True,
        max_model_len=131072, # Adjust if you need longer context
        # token=os.environ.get("HUGGINGFACE_TOKEN"),
    )

    sampling_params = SamplingParams(
        max_tokens=4096,
        temperature=0, # Greedy decoding (deterministic), change if needed
    )

    print(f"Generating responses for {len(prompts)} prompts...")
    # Generate
    outputs = llm.generate(prompts, sampling_params)

    results = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        token_ids = output.outputs[0].token_ids
        
        # Replicate the splitting logic from original script
        # The original script used: output_ids[::-1].index(151668)
        # 151668 seems to be a separator token.
        
        try:
            # Find the separator from the end
            r_index = token_ids[::-1].index(151668)
            idx = len(token_ids) - r_index
            
            thinking_ids = token_ids[:idx]
            answer_ids = token_ids[idx:]
            
            thinking = tokenizer.decode(thinking_ids, skip_special_tokens=True).strip()
            answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
        except ValueError:
            # Separator not found
            thinking = ""
            answer = generated_text.strip()

        results.append({
            "prompt": prompt,
            "thinking": thinking,
            "answer": answer
        })

    pd.DataFrame(results).to_json(args.output, orient="records", lines=True)
    print(f"Done! Wrote {len(results)} results to {args.output}")

if __name__ == "__main__":
    main()
