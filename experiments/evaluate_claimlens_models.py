import sqlite3
import pandas as pd
import os
import ast
from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm
import sys
import numpy as np

# Set up paths
# Assuming script is run from supreme-court-dataset/experiments/
DB_PATH = '../data_store/claimlens/congress.db'
CLAIMS_PATH = '../data_store/claimlens/vote-claims.csv'
MODEL_STORE_PATH = '../store/model_testing'

def load_bills(db_path):
    print(f"Loading bills from {db_path}...")
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return [], []
        
    conn = sqlite3.connect(db_path)
    
    query = """
    SELECT 
        BillCongress, 
        BillType, 
        BillNumber, 
        BillDescription, 
        BillSummary, 
        BillSubjects 
    FROM Bills
    """
    
    try:
        df = pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"Error reading from database: {e}")
        conn.close()
        return [], []
        
    conn.close()
    
    print(f"Loaded {len(df)} bills.")
    
    ids = []
    texts = []
    
    # Pre-calculate to avoid overhead in loop if possible, but loop is fine
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing Bills"):
        try:
            congress = int(row['BillCongress'])
            number = int(row['BillNumber'])
            bill_type = row['BillType']
            bill_id = f"{congress} {bill_type} {number}"
        except (ValueError, TypeError):
            bill_id = f"{row['BillCongress']} {row['BillType']} {row['BillNumber']}"
            
        ids.append(bill_id)
        
        text_parts = []
        if row['BillDescription']:
            text_parts.append(str(row['BillDescription']))
        if row['BillSummary']:
            text_parts.append(str(row['BillSummary']))
        if row['BillSubjects']:
            text_parts.append(str(row['BillSubjects']))
            
        full_text = " ".join(text_parts)
        texts.append(full_text)
        
    return ids, texts

def load_claims(csv_path):
    print(f"Loading claims from {csv_path}...")
    if not os.path.exists(csv_path):
        print(f"Error: CSV not found at {csv_path}")
        return [], []

    df = pd.read_csv(csv_path)
    
    claims = []
    ground_truth = [] 
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading Claims"):
        claim_text = row['claim_entered']
        sourced_bills_raw = row['sourced_bills']
        
        try:
            if isinstance(sourced_bills_raw, str):
                bill_ids = ast.literal_eval(sourced_bills_raw)
            else:
                bill_ids = []
                
            if claim_text and bill_ids:
                claims.append(claim_text)
                ground_truth.append(set(bill_ids))
                
        except Exception as e:
            print(f"Error parsing row {idx}: {e}")
            
    print(f"Loaded {len(claims)} valid claims.")
    return claims, ground_truth

def evaluate_model(model_name_or_path, corpus_texts, corpus_ids, query_texts, ground_truth, k_values=[1, 5, 10]):
    # Suppress output for cleaner logs if needed, but progress bar is useful
    
    # Determine batch size
    if "qwen" in str(model_name_or_path).lower():
        batch_size = 2
    else:
        batch_size = 8192*2*2*2
    print(f"Using batch size: {batch_size}")
    
    try:
        model = SentenceTransformer(model_name_or_path, trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load model {model_name_or_path}: {e}")
        return None

    # Encode Corpus
    print("Encoding corpus...")
    try:
        # Use 4 GPUs
        pool = model.start_multi_process_pool(target_devices=['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3','cuda:4', 'cuda:5', 'cuda:6'])
        
        # Chunking for progress bar
        chunk_size = 8192 
        corpus_embeddings_list = []
        
        # Create chunks
        chunks = [corpus_texts[i:i + chunk_size] for i in range(0, len(corpus_texts), chunk_size)]
        
        for chunk in tqdm(chunks, desc="Encoding Corpus (Multi-GPU)"):
            emb = model.encode_multi_process(chunk, pool, batch_size=batch_size)
            corpus_embeddings_list.append(emb)
            
        model.stop_multi_process_pool(pool)
        
        # Combine embeddings
        if len(corpus_embeddings_list) > 0:
            corpus_embeddings = np.concatenate(corpus_embeddings_list, axis=0)
        else:
            corpus_embeddings = np.array([])
        
        # Convert to tensor and move to GPU for search
        corpus_embeddings = torch.tensor(corpus_embeddings).to('cuda')
    except Exception as e:
        print(f"Multi-GPU encoding failed: {e}. Falling back to single GPU.")
        corpus_embeddings = model.encode(corpus_texts, batch_size=batch_size, show_progress_bar=True, convert_to_tensor=True)
    
    # Encode Queries
    query_embeddings = model.encode(query_texts, batch_size=batch_size, show_progress_bar=True, convert_to_tensor=True)
    
    # Ensure query_embeddings is on the same device as corpus_embeddings
    if corpus_embeddings.device.type == 'cuda':
        query_embeddings = query_embeddings.to('cuda')
    
    # Search
    max_k = max(k_values)
    results = util.semantic_search(query_embeddings, corpus_embeddings, top_k=max_k)
    
    metrics = {k: 0.0 for k in k_values}
    
    for query_idx, hits in enumerate(results):
        gold_ids = ground_truth[query_idx]
        retrieved_ids = [corpus_ids[hit['corpus_id']] for hit in hits]
        
        for k in k_values:
            top_k_retrieved = set(retrieved_ids[:k])
            intersection = len(top_k_retrieved.intersection(gold_ids))
            recall = intersection / len(gold_ids) if len(gold_ids) > 0 else 0
            metrics[k] += recall
            
    for k in k_values:
        metrics[k] /= len(query_texts)
        
    return metrics

def main():
    # Load Data
    bill_ids, bill_texts = load_bills(DB_PATH)
    claims, ground_truth = load_claims(CLAIMS_PATH)
    
    if not bill_ids or not claims:
        print("Failed to load data. Exiting.")
        return

    # Define Models and their trained versions
    models_config = {
        "BAAI/bge-base-en-v1.5": [
            ("1 epoch", os.path.join(MODEL_STORE_PATH, "baai_1")),
            ("3 epochs", os.path.join(MODEL_STORE_PATH, "baai_3")),
            ("5 epochs", os.path.join(MODEL_STORE_PATH, "baai_5")),
        ],
        # "Qwen/Qwen3-Embedding-0.6B": [
        #     ("1 epoch", os.path.join(MODEL_STORE_PATH, "qwen_0.6_1")),
        #     ("3 epochs", os.path.join(MODEL_STORE_PATH, "qwen_0.6_3")),
        # ],
        "sentence-transformers/all-MiniLM-L6-v2": [
            ("1 epoch", os.path.join(MODEL_STORE_PATH, "minilm_1")),
            ("3 epochs", os.path.join(MODEL_STORE_PATH, "minilm_3")),
            ("5 epochs", os.path.join(MODEL_STORE_PATH, "minilm_5")),
        ]
    }
    
    k_values = [1, 5, 10]
    
    for base_model_name, trained_versions in models_config.items():
        print(f"\n{base_model_name}")
        
        # Evaluate Base Model
        # print(f"Evaluating base model: {base_model_name}...")
        base_metrics = evaluate_model(base_model_name, bill_texts, bill_ids, claims, ground_truth, k_values)
        
        if not base_metrics:
            print("Skipping due to base model failure.")
            continue
            
        # Evaluate Trained Versions
        for label, model_path in trained_versions:
            if not os.path.exists(model_path):
                # print(f"Warning: Path {model_path} does not exist. Skipping.")
                continue
                
            print(f"{label}:")
            trained_metrics = evaluate_model(model_path, bill_texts, bill_ids, claims, ground_truth, k_values)
            
            if trained_metrics:
                for k in k_values:
                    base_val = base_metrics[k]
                    trained_val = trained_metrics[k]
                    diff = trained_val - base_val
                    print(f"Recall@{k}: {base_val:.4f} -> {trained_val:.4f} ({diff:+.4f})")
            print(" ") 

if __name__ == "__main__":
    main()
