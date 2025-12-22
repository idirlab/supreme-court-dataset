import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import argparse
import os

def get_embeddings(model, tokenizer, texts, batch_size=32, device="cuda"):
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            # Assuming mean pooling for embedding model; adjust if specific instruction exists for Qwen3-Embedding
            # Many modern embedding models use the last hidden state of the EOS token or mean pooling.
            # We will use mean pooling as a safe default for "Embedding" models unless it's a decoder-only used as encoder.
            # Qwen is decoder-only usually.
            # If it's an embedding model, it might be trained to use the last token.
            # Let's try to use the last hidden state mean.
            last_hidden_states = outputs.last_hidden_state
            embeddings = last_hidden_states.mean(dim=1)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu())
    return torch.cat(all_embeddings, dim=0)

def main():
    parser = argparse.ArgumentParser(description="Check semantic similarity between claims")
    parser.add_argument("--input", type=str, default="../claims_raw.csv", help="Input CSV file")
    parser.add_argument("--output", type=str, default="../results/similarity_report.csv", help="Output CSV file")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-Embedding-8B", help="HuggingFace model name")
    parser.add_argument("--threshold", type=float, default=0.85, help="Similarity threshold")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    args = parser.parse_args()

    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)

    # Fix for filtered_claims.csv which uses 'case_name' instead of 'name'
    if "name" not in df.columns and "case_name" in df.columns:
        print("Renaming 'case_name' to 'name'...")
        df["name"] = df["case_name"]

    # Fix for missing 'docket' column
    if "docket" not in df.columns:
        print("Docket column missing. Attempting to load from ../clean_data_with_details.csv...")
        meta_path = "../clean_data_with_details.csv"
        if os.path.exists(meta_path):
            try:
                meta_df = pd.read_csv(meta_path)
                if "name" in meta_df.columns and "docket" in meta_df.columns:
                    name_to_docket = meta_df.set_index("name")["docket"].to_dict()
                    df["docket"] = df["name"].map(name_to_docket)
                    print("Populated 'docket' column from metadata.")
            except Exception as e:
                print(f"Warning: Could not populate dockets: {e}")
        else:
            print(f"Warning: Metadata file not found at {meta_path}")
    
    # Ensure we have a claim column
    if "claim" not in df.columns:
        print("Error: 'claim' column not found in input CSV.")
        return
    
    claims = df["claim"].fillna("").tolist()
    # Filter empty claims
    valid_indices = [i for i, c in enumerate(claims) if c.strip()]
    valid_claims = [claims[i] for i in valid_indices]
    
    print(f"Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # Use bfloat16 for H100 efficiency
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Simple DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    
    model.to(device)
    model.eval()
    
    print(f"Generating embeddings for {len(valid_claims)} claims...")
    embeddings = get_embeddings(model, tokenizer, valid_claims, batch_size=args.batch_size, device=device)
    
    print("Computing similarity matrix...")
    
    results = []
    # Compute similarity in chunks to show progress and save memory
    # We iterate over rows in chunks
    chunk_size = args.batch_size
    
    for i in tqdm(range(0, len(embeddings), chunk_size), desc="Computing similarity"):
        end = min(i + chunk_size, len(embeddings))
        batch_embeddings = embeddings[i:end].to(device)
        
        # (B, D) @ (D, N) -> (B, N)
        # Note: embeddings is on CPU, we need to move target to GPU or keep it there?
        # If N is large, embeddings might not fit on GPU.
        # Let's assume we move the whole embeddings to GPU if it fits, or chunk the target too.
        # For simplicity, let's try moving full embeddings to GPU if possible, or keep on CPU and do CPU mm (slow).
        # Better: move target chunk to GPU if needed.
        # But to compute against ALL, we need all on GPU.
        # If N=100k, 20GB. H100 has 80GB. It fits.
        pass

    # Move all embeddings to device for fast MM if it fits
    try:
        embeddings_device = embeddings.to(device)
    except RuntimeError:
        print("Warning: Embeddings too large for GPU, using CPU for similarity (slow).")
        embeddings_device = embeddings
        device = "cpu"

    for i in tqdm(range(0, len(embeddings), chunk_size), desc="Computing similarity"):
        end = min(i + chunk_size, len(embeddings))
        batch_embeddings = embeddings_device[i:end]
        
        # Compute similarity
        sim_batch = torch.mm(batch_embeddings, embeddings_device.t())
        
        # Filter by threshold
        # We only want upper triangle: col > row
        # absolute row index = i + row_in_batch
        
        mask = sim_batch > args.threshold
        batch_indices = torch.nonzero(mask)
        
        if batch_indices.numel() == 0:
            continue
            
        row_in_batch = batch_indices[:, 0]
        col_indices = batch_indices[:, 1]
        
        abs_rows = row_in_batch + i
        
        # Upper triangle check
        valid_mask = col_indices > abs_rows
        
        valid_rows = abs_rows[valid_mask]
        valid_cols = col_indices[valid_mask]
        valid_scores = sim_batch[row_in_batch[valid_mask], valid_cols]
        
        # Collect results
        for r, c, score in zip(valid_rows.cpu().tolist(), valid_cols.cpu().tolist(), valid_scores.cpu().tolist()):
            idx1 = valid_indices[r]
            idx2 = valid_indices[c]
            
            row1 = df.iloc[idx1]
            row2 = df.iloc[idx2]
            
            results.append({
                "index1": idx1,
                "index2": idx2,
                "claim1": row1["claim"],
                "claim2": row2["claim"],
                "score": score,
                "name1": row1.get("name", ""),
                "name2": row2.get("name", ""),
                "docket1": row1.get("docket", ""),
                "docket2": row2.get("docket", "")
            })

    print(f"Found {len(results)} pairs with similarity > {args.threshold}")
    
    out_df = pd.DataFrame(results)
    
    # Sort by score descending so top is most similar
    out_df = out_df.sort_values(by="score", ascending=False)
    
    # Print top 5 most similar
    print("\nTop 5 most similar pairs:")
    if not out_df.empty:
        cols_to_show = ["name1", "claim1", "name2", "claim2", "score"]
        print(out_df.head(5)[cols_to_show].to_string(index=False))

    out_df.to_csv(args.output, index=False)
    print(f"Saved sorted report to {args.output}")

if __name__ == "__main__":
    main()
