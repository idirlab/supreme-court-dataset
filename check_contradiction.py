import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import argparse
import itertools

def check_contradictions(model, tokenizer, pairs, batch_size=32, device="cuda", save_all=False):
    results = []
    # Labels for textattack/bert-base-uncased-MNLI:
    # 0: contradiction, 1: entailment, 2: neutral (usually, but let's verify with model.config.id2label)
    # Actually, for this specific model: 0: contradiction, 1: entailment, 2: neutral is common for MNLI.
    # We will check model.config.id2label after loading.

    # Handle DataParallel wrapper to access config
    if hasattr(model, "module"):
        config = model.module.config
    else:
        config = model.config
    
    for i in tqdm(range(0, len(pairs), batch_size), desc="Checking contradictions"):
        batch_pairs = pairs[i:i+batch_size]
        premises = [p[0] for p in batch_pairs]
        hypotheses = [p[1] for p in batch_pairs]
        
        inputs = tokenizer(premises, hypotheses, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            predictions = torch.argmax(probs, dim=1)
            
        # Move to CPU
        probs = probs.cpu().numpy()
        predictions = predictions.cpu().numpy()
        
        for j, pred in enumerate(predictions):
            # We need to map pred index to label
            label = config.id2label[pred].lower()
            score = probs[j][pred]
            
            # If save_all is True, save everything. Otherwise, only save contradictions.
            if save_all or "contradiction" in label:
                results.append({
                    "claim1": premises[j],
                    "claim2": hypotheses[j],
                    "label": label,
                    "score": float(score),
                    "pair_info": batch_pairs[j][2] # (index1, index2, name1, name2)
                })
    return results

def main():
    parser = argparse.ArgumentParser(description="Check contradictions between claims")
    parser.add_argument("--input", type=str, default="results_80B_claims.csv", help="Input CSV file")
    parser.add_argument("--output", type=str, default="contradiction_report.csv", help="Output CSV file")
    parser.add_argument("--model", type=str, default="textattack/bert-base-uncased-MNLI", help="BERT-based NLI model")
    parser.add_argument("--batch_size", type=int, default=65536, help="Batch size")
    parser.add_argument("--global_check", action="store_true", help="Check every claim against every other claim globally")
    args = parser.parse_args()

    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    
    if "claim" not in df.columns or "name" not in df.columns:
        print("Error: 'claim' and 'name' columns required.")
        return

    all_pairs = []
    
    if args.global_check:
        print("Generating GLOBAL pairs (checking every claim against every other claim)...")
        # Global check: all combinations of all claims
        claims = df["claim"].dropna().tolist()
        indices = df.index.tolist()
        names = df["name"].tolist()
        
        # Create list of (index, claim, name) tuples
        items = list(zip(indices, claims, names))
        
        # Use combinations to avoid self-pairs and duplicates (A-B same as B-A for iteration, though NLI is asymmetric)
        # If we want strict global check, we might want permutations, but combinations is usually enough for "overlap" check.
        # Let's stick to combinations for N^2/2 complexity.
        total_pairs = (len(items) * (len(items) - 1)) // 2
        print(f"Estimated pairs: {total_pairs}")
        
        # We can't pre-generate all pairs in memory if N is huge (e.g. 100k claims -> 5B pairs).
        # If N is small (e.g. < 5000), it's fine.
        # Assuming N is manageable for now given the user request.
        if len(items) > 10000:
            print("Warning: Global check with >10k claims will generate >50M pairs. This might be slow or OOM.")
        
        for (i1, c1, n1), (i2, c2, n2) in tqdm(itertools.combinations(items, 2), total=total_pairs, desc="Generating pairs"):
            all_pairs.append((c1, c2, (i1, i2, n1, n2)))
            
    else:
        # Within-case check
        print("Generating pairs within cases...")
        grouped = df.groupby("name")
        # Use tqdm for progress on pair generation
        for name, group in tqdm(grouped, desc="Generating pairs"):
            claims = group["claim"].dropna().tolist()
            indices = group.index.tolist()
            
            if len(claims) < 2:
                continue
                
            for (i1, c1), (i2, c2) in itertools.combinations(zip(indices, claims), 2):
                # For within-case, name1 and name2 are the same
                all_pairs.append((c1, c2, (i1, i2, name, name)))

    if not all_pairs:
        print("No pairs found.")
        return

    print(f"Total pairs to check: {len(all_pairs)}")
    
    print(f"Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    
    model.to(device)
    model.eval()
    
    # If global check -> only save contradictions (save_all=False)
    # If within-case -> save all pairs (save_all=True)
    save_all_pairs = not args.global_check
    
    contradictions = check_contradictions(model, tokenizer, all_pairs, batch_size=args.batch_size, device=device, save_all=save_all_pairs)
    
    print(f"Found {len(contradictions)} results.")
    
    # Format output
    out_rows = []
    for c in contradictions:
        idx1, idx2, name1, name2 = c["pair_info"]
        out_rows.append({
            "name1": name1,
            "name2": name2,
            "index1": idx1,
            "index2": idx2,
            "claim1": c["claim1"],
            "claim2": c["claim2"],
            "label": c["label"],
            "score": c["score"]
        })
        
    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(args.output, index=False)
    print(f"Saved report to {args.output}")

if __name__ == "__main__":
    main()
