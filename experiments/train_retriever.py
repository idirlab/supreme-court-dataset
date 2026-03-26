import pandas as pd
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
from datasets import Dataset
import os
import ast
import argparse
import torch

def load_data(train_path, cases_path, test_path):
    print(f"Loading training data from {train_path}...")
    train_df = pd.read_csv(train_path)
    
    print(f"Loading cases from {cases_path}...")
    cases_df = pd.read_csv(cases_path)
    
    print(f"Loading test data from {test_path}...")
    test_df = pd.read_csv(test_path)
    
    return train_df, cases_df, test_df

def prepare_training_dataset(train_df, cases_df):
    print("Preparing training dataset...")
    
    # Create a mapping from case name to full text
    # We'll combine facts, api_question, and api_conclusion for a rich representation
    case_text_map = {}
    for idx, row in cases_df.iterrows():
        name = row['name']
        text_parts = [
            str(row.get('facts', '')),
            str(row.get('api_question', '')),
            str(row.get('api_conclusion', ''))
        ]
        full_text = " ".join([t for t in text_parts if t and t != 'nan'])
        case_text_map[name] = full_text
        
    anchors = []
    positives = []
    skipped = 0
    
    for idx, row in train_df.iterrows():
        claim = row['claim']
        case_name_raw = row['case_name']
        
        try:
            # Handle potential list format in string
            if isinstance(case_name_raw, str) and case_name_raw.startswith('[') and case_name_raw.endswith(']'):
                case_names = ast.literal_eval(case_name_raw)
            else:
                case_names = [case_name_raw]
                
            # For training, we'll create a pair for each valid case associated with the claim
            for case_name in case_names:
                if case_name in case_text_map:
                    anchors.append(claim)
                    positives.append(case_text_map[case_name])
                else:
                    # Try normalizing or fuzzy matching if needed, but for now skip
                    # print(f"Warning: Case '{case_name}' not found in cases file.")
                    skipped += 1
                    
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            skipped += 1
            
    print(f"Created {len(anchors)} training pairs. Skipped {skipped} case references.")
    
    dataset = Dataset.from_dict({
        "anchor": anchors,
        "positive": positives
    })
    
    return dataset, case_text_map

def train_model(train_dataset, output_path, num_epochs=1, batch_size=16, model_name='BAAI/bge-base-en-v1.5'):
    print(f"Initializing model {model_name}...")
    model = SentenceTransformer(model_name)
    loss = losses.MultipleNegativesRankingLoss(model)
    
    args = SentenceTransformerTrainingArguments(
        output_dir=output_path,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=True, # Use mixed precision
        logging_steps=100,
        save_strategy="epoch",
        eval_strategy="no",
        report_to="none"
    )
    
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
    )
    
    print(f"Starting training on {torch.cuda.device_count()} GPUs...")
    trainer.train()
    
    print(f"Saving model to {output_path}...")
    model.save_pretrained(output_path)
    return model

def evaluate_model(model, test_df, case_text_map, k_values=[1, 5, 10]):
    print("Evaluating model...")
    
    # Prepare corpus (all cases)
    corpus = case_text_map
    corpus_ids = list(corpus.keys())
    corpus_embeddings = model.encode([corpus[cid] for cid in corpus_ids], show_progress_bar=True, convert_to_tensor=True)
    
    # Prepare queries (test claims)
    queries = []
    relevant_docs = {} # query_idx -> set of relevant case_ids
    
    valid_test_rows = 0
    for idx, row in test_df.iterrows():
        claim = row['claim']
        case_name_raw = row['case_name']
        
        try:
            if isinstance(case_name_raw, str) and case_name_raw.startswith('[') and case_name_raw.endswith(']'):
                case_names = ast.literal_eval(case_name_raw)
            else:
                case_names = [case_name_raw] if pd.notna(case_name_raw) else []
            
            # Filter to cases that exist in our corpus
            valid_cases = [c for c in case_names if c in case_text_map]
            
            if valid_cases:
                queries.append(claim)
                relevant_docs[valid_test_rows] = set(valid_cases)
                valid_test_rows += 1
                
        except Exception as e:
            print(f"Error processing test row {idx}: {e}")

    print(f"Evaluating on {len(queries)} test queries against {len(corpus_ids)} documents.")
    
    # Encode queries
    query_embeddings = model.encode(queries, show_progress_bar=True, convert_to_tensor=True)
    
    # Perform search
    # We want top K results
    from sentence_transformers.util import semantic_search
    
    # Search for max K
    max_k = max(k_values)
    results = semantic_search(query_embeddings, corpus_embeddings, top_k=max_k)
    
    # Calculate metrics
    metrics = {f"Recall@{k}": 0.0 for k in k_values}
    
    for query_idx, hits in enumerate(results):
        gold_cases = relevant_docs[query_idx]
        
        # Get retrieved case names
        retrieved_cases = [corpus_ids[hit['corpus_id']] for hit in hits]
        
        for k in k_values:
            top_k_retrieved = set(retrieved_cases[:k])
            # Check if ANY relevant case is retrieved (Recall@K for retrieval usually means "is the answer in top K")
            # Or strictly, recall = intersection / total_relevant
            
            # Let's compute standard Recall@K: |relevant_retrieved| / |total_relevant|
            intersection = len(top_k_retrieved.intersection(gold_cases))
            recall = intersection / len(gold_cases) if len(gold_cases) > 0 else 0
            metrics[f"Recall@{k}"] += recall
            
    # Average
    for k in k_values:
        metrics[f"Recall@{k}"] /= len(queries)
        
    print("Evaluation Results:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a retriever model on legal claims.")
    parser.add_argument("--train_file", default="../dataset/train_set.csv", help="Path to training CSV")
    parser.add_argument("--cases_file", default="../dataset/supreme_court_cases.csv", help="Path to cases CSV")
    parser.add_argument("--test_file", default="../dataset/test_set.csv", help="Path to test CSV")
    parser.add_argument("--output_dir", default="../data_store/output/retriever_model", help="Directory to save trained model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--model_name", default="Qwen/Qwen3-Embedding-0.6B", help="Base model name")
    
    args = parser.parse_args()
    
    train_df, cases_df, test_df = load_data(args.train_file, args.cases_file, args.test_file)
    
    train_dataset, case_text_map = prepare_training_dataset(train_df, cases_df)
    
    # Evaluate base model first
    print("\n--- Evaluating Base Model (Pre-training) ---")
    base_model = SentenceTransformer(args.model_name)
    base_metrics = evaluate_model(base_model, test_df, case_text_map)
    
    # Train
    print("\n--- Training Model ---")
    model = train_model(train_dataset, args.output_dir, args.epochs, args.batch_size, args.model_name)
    
    # Evaluate trained
    print("\n--- Evaluating Trained Model ---")
    trained_metrics = evaluate_model(model, test_df, case_text_map)
    
    # Print comparison
    print("\n--- Comparison (Base vs Trained) ---")
    for k in base_metrics:
        base_val = base_metrics[k]
        trained_val = trained_metrics[k]
        diff = trained_val - base_val
        print(f"{k}: {base_val:.4f} -> {trained_val:.4f} ({diff:+.4f})")
