import argparse
import json
import os
import pandas as pd
from typing import List
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

def parse_args():
    parser = argparse.ArgumentParser(description="Enhanced batch inference for large MoE models with vLLM")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--prompts", type=str, default="prompts.jsonl", help="Path to prompts JSONL (OpenAI format)")
    parser.add_argument("--output", type=str, default="outputs.jsonl", help="Output file (JSONL)")
    parser.add_argument("--max_tokens", type=int, default=262144, help="Max new tokens to generate per prompt")
    parser.add_argument("--max_model_len", type=int, default=None, help="Max model (context) length; if unset falls back to max_tokens")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs (tensor parallel size)")
    parser.add_argument("--devices", type=str, required=True, help="Comma-separated CUDA device IDs, e.g. 0,1,2,3")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"], help="Weights dtype")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85, help="Memory utilization target")
    return parser.parse_args()

def load_prompts(path: str, tokenizer) -> List[str]:
    df = pd.read_json(path, lines=True)
    prompts = []
    for _, row in df.iterrows():
        prompt = tokenizer.apply_chat_template(row["body"]["messages"], tokenize=False)
        prompts.append(prompt)
    return prompts

def main():
    args = parse_args()
    assert args.gpus == len(args.devices.split(",")), "Number of GPUs must match number of devices specified"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = load_prompts(args.prompts, tokenizer)

    max_model_len = args.max_model_len if args.max_model_len is not None else args.max_tokens
    if max_model_len is None:
        # Safe default if neither is set
        max_model_len = 8192

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.gpus,
        max_model_len=max_model_len,
        enable_expert_parallel=True,  # keep EP on for MoE
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        quantization="fp8",
    )
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    outputs = llm.generate(prompts, sampling_params)
    results = []
    for prompt, output in zip(prompts, outputs):
        results.append({"prompt": prompt, "output": output})#.outputs[0].text

    pd.DataFrame(results).to_json(args.output, orient="records", lines=True)

if __name__ == "__main__":
    main()