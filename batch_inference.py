import argparse
import json
import os
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--prompts", type=str, required=True, help="Path to prompts JSONL (OpenAI format)")
    parser.add_argument("--output", type=str, default="outputs.jsonl", help="Output file")
    parser.add_argument("--max_tokens", type=int, default=10000)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--devices", type=str, required=True, help="Comma-separated list of CUDA device IDs, e.g. 4,5,6,7")
    return parser.parse_args()

def load_prompts(path, tokenizer):
    df = pd.read_json(path, lines=True)
    prompts = []
    for _, row in df.iterrows():
        # Use HuggingFace chat template for Llama 3.3
        prompt = tokenizer.apply_chat_template(row['body']["messages"], tokenize=False)
        prompts.append(prompt)
    return prompts

# def main():
args = parse_args()

assert args.gpus == len(args.devices.split(",")), "Number of GPUs must match number of devices specified"

os.environ["CUDA_VISIBLE_DEVICES"] = args.devices

# Load tokenizer for chat template
tokenizer = AutoTokenizer.from_pretrained(args.model)

# Load prompts
prompts = load_prompts(args.prompts, tokenizer)

# Setup vllm
llm = LLM(model=args.model, tensor_parallel_size=args.gpus, max_model_len=args.max_tokens)
sampling_params = SamplingParams(
    max_tokens=args.max_tokens,
    temperature=args.temperature,
    top_p=args.top_p
)

# Inference
outputs = llm.generate(prompts, sampling_params)
results = []
for prompt, output in zip(prompts, outputs):
    results.append({
        "prompt": prompt,
        "output": output.outputs[0].text
    })

# Save results
pd.DataFrame(results).to_json(args.output, orient="records", lines=True)
 