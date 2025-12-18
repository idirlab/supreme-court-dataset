
# pip install accelerate transformers torch pandas pillow tqdm
import json
from tqdm import tqdm
from accelerate import PartialState

import torch
import pandas as pd
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

distributed_state = PartialState()

model_id = "google/gemma-3-27b-it"
prompt_path = "overlap_prompts.jsonl"    

model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map=distributed_state.device
).eval()

processor = AutoProcessor.from_pretrained(model_id)
tokenizer = processor.tokenizer    

def load_prompts(path, tokenizer):
    df = pd.read_json(path, lines=True)
    prompts = []
    for _, row in df.iterrows():
        prompt = tokenizer.apply_chat_template(
            row['body']["messages"],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)
    return prompts



prompts = load_prompts(prompt_path, tokenizer)

with distributed_state.split_between_processes(prompts) as prompts_split:
    results = []   
    for prompt in tqdm(prompts_split, disable=not distributed_state.is_local_main_process):
        model_inputs = tokenizer(
            [prompt],
            return_tensors="pt",
        ).to(model.device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=4096,
        )
        output_ids = generated_ids[0][model_inputs.input_ids.shape[-1]:].tolist()
        try:
            idx = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            idx = 0

        thinking = tokenizer.decode(output_ids[:idx], skip_special_tokens=True).strip()
        answer = tokenizer.decode(output_ids[idx:], skip_special_tokens=True).strip()

        results.append({
            "prompt": prompt,
            "thinking": thinking,
            "answer": answer
        })
    
    output_file = f"overlap_output_rank_{distributed_state.process_index}.jsonl"
    pd.DataFrame(results).to_json(output_file, orient="records", lines=True)
    print(f"Process {distributed_state.process_index} wrote {output_file}")

distributed_state.wait_for_everyone()
if distributed_state.is_main_process:
    print("All processes done.")

 