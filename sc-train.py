import pandas as pd
import os
import numpy as np
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from tqdm import tqdm
import random

class SemanticSimilarityTrainer:
    def __init__(self, csv_file='sc-claims_v3.csv', model_name='Qwen/Qwen3-Embedding-0.6B'):
        """
        Initialize the semantic similarity trainer
        
        Args:
            csv_file: Path to the CSV file
            model_name: Base sentence transformer model to use
        """
        self.csv_file = csv_file
        self.model_name = model_name
        
        # Force single GPU usage to avoid multi-GPU issues
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            # Set CUDA_VISIBLE_DEVICES to use only first GPU
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        else:
            self.device = torch.device('cpu')
            
        print(f"Using device: {self.device}")
        
        # Load and prepare data
        self.df = pd.read_csv(csv_file)
        self.prepare_data()
        
    def prepare_data(self):
        """Prepare the training data"""
        print("Preparing data...")
        
        # Remove rows with missing values in key columns
        required_columns = ['name'] + [f'claim_{i}' for i in range(1, 6)]
        initial_count = len(self.df)
        self.df = self.df.dropna(subset=required_columns)
        
        # Also remove rows where any claim column is empty string or just whitespace
        for claim_col in [f'claim_{i}' for i in range(1, 6)]:
            self.df = self.df[self.df[claim_col].astype(str).str.strip() != '']
        
        # Create mapping from case name to index
        self.name_to_idx = {name: idx for idx, name in enumerate(self.df['name'].values)}
        self.case_names = self.df['name'].values
        
        removed_count = initial_count - len(self.df)
        print(f"Removed {removed_count} rows with missing or empty claim values")
        print(f"Dataset contains {len(self.df)} cases")
        
    def create_training_examples(self, claim_columns=['claim_1', 'claim_2', 'claim_3', 'claim_4']):
        """
        Create training examples using multiple claim columns
        
        Args:
            claim_columns: List of claim columns to use for training
        """
        print(f"Creating training examples using {claim_columns}...")
        
        train_examples = []
        
        for claim_column in claim_columns:
            print(f"Processing {claim_column}...")
            
            for idx, row in self.df.iterrows():
                case_name = row['name']
                claim_text = row[claim_column]
                
                # Skip if claim text is missing, NaN, or empty
                if pd.isna(claim_text) or pd.isna(case_name) or str(claim_text).strip() == '':
                    continue
                    
                # Positive example: claim should match its case name
                train_examples.append(InputExample(
                    texts=[str(claim_text), str(case_name)], 
                    label=1.0
                ))
                
                # Negative examples: claim should not match random other case names
                # Sample 2 negative examples per positive example (reduced to balance dataset)
                other_names = [name for name in self.case_names if name != case_name]
                negative_samples = random.sample(other_names, min(2, len(other_names)))
                
                for neg_name in negative_samples:
                    train_examples.append(InputExample(
                        texts=[str(claim_text), str(neg_name)], 
                        label=0.0
                    ))
        
        print(f"Created {len(train_examples)} training examples")
        return train_examples
    
    def train_model(self, claim_columns=['claim_1', 'claim_2', 'claim_3', 'claim_4'], epochs=4, batch_size=8):
        """
        Train the semantic similarity model
        
        Args:
            claim_columns: List of claim columns to use for training
            epochs: Number of training epochs
            batch_size: Training batch size
        """
        print("Initializing model...")
        # Force single GPU usage to avoid DataParallel issues
        if torch.cuda.is_available():
            device_str = 'cuda:0'  # Use first GPU
        else:
            device_str = 'cpu'
        
        self.model = SentenceTransformer(self.model_name, device=device_str)
        
        # Create training examples
        train_examples = self.create_training_examples(claim_columns)
        
        # Create data loader
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        
        # Define loss function
        train_loss = losses.CosineSimilarityLoss(self.model)
        
        print(f"Training model for {epochs} epochs...")
        torch.cuda.empty_cache()

        
        # Train the model with single GPU to avoid DataParallel issues
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=100,
            show_progress_bar=True,
            use_amp=False,  # Disable automatic mixed precision
            checkpoint_path=None,  # Disable checkpointing
            checkpoint_save_steps=None
        )
        
        print("Training completed!")
        
    def evaluate_model(self, test_claim_columns=['claim_5'], model_name="Trained Model", save_predictions=True):
        """
        Evaluate the model using recall@1, recall@5, and recall@10
        
        Args:
            test_claim_columns: List of claim columns to use for evaluation
            model_name: Name to display in results (for distinguishing trained vs naive)
            save_predictions: Whether to save detailed predictions to CSV
        """
        print(f"Evaluating {model_name}...")
        
        # Encode all case names
        print("Encoding case names...")
        case_embeddings = self.model.encode(self.case_names, convert_to_tensor=True)
        
        total_recall_1 = 0
        total_recall_5 = 0
        total_recall_10 = 0
        total_queries = 0
        
        # Store all predictions for CSV export
        all_predictions = []
        
        for claim_col in test_claim_columns:
            print(f"Evaluating on {claim_col}...")
            
            recall_1_scores = []
            recall_5_scores = []
            recall_10_scores = []
            
            for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc=f"{model_name} - {claim_col}"):
                claim_text = row[claim_col]
                true_case_name = row['name']
                
                # Skip if claim text is missing, NaN, or empty
                if pd.isna(claim_text) or str(claim_text).strip() == '':
                    continue
                
                # Encode the query claim
                query_embedding = self.model.encode([str(claim_text)], convert_to_tensor=True)
                
                # Compute similarities
                similarities = cosine_similarity(
                    query_embedding.cpu().numpy(), 
                    case_embeddings.cpu().numpy()
                )[0]
                
                # Get top-k indices
                top_indices = np.argsort(similarities)[::-1]
                
                # Find the rank of the true case
                true_idx = self.name_to_idx[true_case_name]
                true_rank = np.where(top_indices == true_idx)[0][0] + 1  # 1-indexed rank
                
                # Calculate recall scores
                recall_1 = 1.0 if true_rank <= 1 else 0.0
                recall_5 = 1.0 if true_rank <= 5 else 0.0
                recall_10 = 1.0 if true_rank <= 10 else 0.0
                
                recall_1_scores.append(recall_1)
                recall_5_scores.append(recall_5)
                recall_10_scores.append(recall_10)
                
                # Store prediction details for CSV
                if save_predictions:
                    # Get top 10 predictions
                    top_10_indices = top_indices[:10]
                    top_10_names = [self.case_names[i] for i in top_10_indices]
                    top_10_similarities = [similarities[i] for i in top_10_indices]
                    
                    prediction_row = {
                        'query_claim': str(claim_text),
                        'true_case_name': true_case_name,
                        'true_rank': true_rank,
                        'recall_1': recall_1,
                        'recall_5': recall_5,
                        'recall_10': recall_10,
                        'claim_column': claim_col,
                        'model_type': model_name.lower().replace(' ', '_')
                    }
                    
                    # Add top 10 predictions
                    for i in range(10):
                        if i < len(top_10_names):
                            prediction_row[f'top_{i+1}_case'] = top_10_names[i]
                            prediction_row[f'top_{i+1}_similarity'] = top_10_similarities[i]
                        else:
                            prediction_row[f'top_{i+1}_case'] = ''
                            prediction_row[f'top_{i+1}_similarity'] = 0.0
                    
                    all_predictions.append(prediction_row)
            
            # Calculate averages for this claim column
            avg_recall_1 = np.mean(recall_1_scores)
            avg_recall_5 = np.mean(recall_5_scores)
            avg_recall_10 = np.mean(recall_10_scores)
            
            print(f"{model_name} - {claim_col} - Recall@1: {avg_recall_1:.4f}, Recall@5: {avg_recall_5:.4f}, Recall@10: {avg_recall_10:.4f}")
            
            total_recall_1 += avg_recall_1 * len(recall_1_scores)
            total_recall_5 += avg_recall_5 * len(recall_5_scores)
            total_recall_10 += avg_recall_10 * len(recall_10_scores)
            total_queries += len(recall_1_scores)
        
        # Calculate overall averages
        overall_recall_1 = total_recall_1 / total_queries
        overall_recall_5 = total_recall_5 / total_queries
        overall_recall_10 = total_recall_10 / total_queries
        
        print(f"\n{model_name} - Overall Results:")
        print(f"Recall@1: {overall_recall_1:.4f}")
        print(f"Recall@5: {overall_recall_5:.4f}")
        print(f"Recall@10: {overall_recall_10:.4f}")
        
        # Save predictions to CSV
        if save_predictions and all_predictions:
            model_type_clean = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
            csv_filename = f"predictions_{model_type_clean}.csv"
            
            predictions_df = pd.DataFrame(all_predictions)
            predictions_df.to_csv(csv_filename, index=False)
            print(f"Predictions saved to {csv_filename}")
        
        return overall_recall_1, overall_recall_5, overall_recall_10
    
    def save_model(self, save_path='sc_semantic_model'):
        """
        Save the trained model and associated data
        
        Args:
            save_path: Directory to save the model
        """
        print(f"Saving model to {save_path}...")
        
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Save the sentence transformer model
        self.model.save(save_path)
        
        # Save additional data
        model_data = {
            'case_names': self.case_names,
            'name_to_idx': self.name_to_idx,
            'model_name': self.model_name
        }
        
        with open(os.path.join(save_path, 'model_data.pkl'), 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved successfully to {save_path}")
    
    def model_exists(self, model_path='sc_semantic_model'):
        """
        Check if a trained model already exists
        
        Args:
            model_path: Path to check for existing model
            
        Returns:
            bool: True if model exists, False otherwise
        """
        return (os.path.exists(model_path) and 
                os.path.exists(os.path.join(model_path, 'model_data.pkl')) and
                os.path.exists(os.path.join(model_path, 'config.json')))
    
    def load_naive_model(self):
        """
        Load the naive (pre-trained, not fine-tuned) model for comparison
        """
        print("Loading naive (pre-trained) model...")
        
        # Force single GPU usage
        if torch.cuda.is_available():
            device_str = 'cuda:1'
            os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        else:
            device_str = 'cpu'
            
        # Load the base model without any fine-tuning
        naive_model = SentenceTransformer(self.model_name, device=device_str)
        return naive_model
        """
        Load a previously trained model
        
        Args:
            load_path: Directory containing the saved model
        """
        print(f"Loading model from {load_path}...")
        
        # Force single GPU usage
        if torch.cuda.is_available():
            device_str = 'cuda:0'
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        else:
            device_str = 'cpu'
        
        # Load the sentence transformer model
        self.model = SentenceTransformer(load_path, device=device_str)
        
    def load_model(self, load_path='sc_semantic_model'):
        """
        Load a previously trained model
        
        Args:
            load_path: Directory containing the saved model
        """
        print(f"Loading trained model from {load_path}...")
        
        # Force single GPU usage
        if torch.cuda.is_available():
            device_str = 'cuda:0'
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        else:
            device_str = 'cpu'
        
        # Load the sentence transformer model
        self.model = SentenceTransformer(load_path, device=device_str)
        
        # Load additional data
        with open(os.path.join(load_path, 'model_data.pkl'), 'rb') as f:
            model_data = pickle.load(f)
        
        self.case_names = model_data['case_names']
        self.name_to_idx = model_data['name_to_idx']
        self.model_name = model_data['model_name']
        
        print("Trained model loaded successfully!")
    
    def evaluate_both_models(self, test_claim_columns=['claim_5']):
        """
        Evaluate both the naive and trained models for comparison
        
        Args:
            test_claim_columns: List of claim columns to use for evaluation
            
        Returns:
            dict: Results for both models
        """
        results = {}
        
        # Evaluate naive model
        print("="*60)
        print("EVALUATING NAIVE (PRE-TRAINED) MODEL")
        print("="*60)
        naive_model = self.load_naive_model()
        original_model = self.model
        self.model = naive_model  # Temporarily switch to naive model
        
        naive_r1, naive_r5, naive_r10 = self.evaluate_model(test_claim_columns, "Naive Model", save_predictions=True)
        results['naive'] = {'recall_1': naive_r1, 'recall_5': naive_r5, 'recall_10': naive_r10}
        
        # Evaluate trained model
        print("\n" + "="*60)
        print("EVALUATING TRAINED (FINE-TUNED) MODEL")
        print("="*60)
        self.model = original_model  # Switch back to trained model
        
        trained_r1, trained_r5, trained_r10 = self.evaluate_model(test_claim_columns, "Trained Model", save_predictions=True)
        results['trained'] = {'recall_1': trained_r1, 'recall_5': trained_r5, 'recall_10': trained_r10}
        
        # Print comparison
        print("\n" + "="*60)
        print("MODEL COMPARISON RESULTS")
        print("="*60)
        print(f"Naive Model    - Recall@1: {naive_r1:.4f}, Recall@5: {naive_r5:.4f}, Recall@10: {naive_r10:.4f}")
        print(f"Trained Model  - Recall@1: {trained_r1:.4f}, Recall@5: {trained_r5:.4f}, Recall@10: {trained_r10:.4f}")
        print(f"Improvement    - Recall@1: {trained_r1-naive_r1:+.4f}, Recall@5: {trained_r5-naive_r5:+.4f}, Recall@10: {trained_r10-naive_r10:+.4f}")
        
        # Calculate relative gains (handle division by zero)
        r1_gain = ((trained_r1/naive_r1-1)*100) if naive_r1 > 0 else float('inf')
        r5_gain = ((trained_r5/naive_r5-1)*100) if naive_r5 > 0 else float('inf')
        r10_gain = ((trained_r10/naive_r10-1)*100) if naive_r10 > 0 else float('inf')
        
        print(f"Relative Gain  - Recall@1: {r1_gain:+.1f}%, Recall@5: {r5_gain:+.1f}%, Recall@10: {r10_gain:+.1f}%")
        
        # Save combined results to CSV
        combined_results = {
            'metric': ['Recall@1', 'Recall@5', 'Recall@10'],
            'naive_model': [naive_r1, naive_r5, naive_r10],
            'trained_model': [trained_r1, trained_r5, trained_r10],
            'improvement': [trained_r1-naive_r1, trained_r5-naive_r5, trained_r10-naive_r10],
            'relative_gain_percent': [r1_gain, r5_gain, r10_gain]
        }
        
        results_df = pd.DataFrame(combined_results)
        results_df.to_csv('model_comparison_results.csv', index=False)
        print(f"\nComparison results saved to model_comparison_results.csv")
        print(f"Detailed predictions saved to predictions_naive_model.csv and predictions_trained_model.csv")
        
        return results

def main():
    """Main training and evaluation pipeline"""
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Initialize trainer
    trainer = SemanticSimilarityTrainer('sc-claims_v4.csv')

    model_path = './ssm/sc_semantic_model_qwen3-0.6B1'

    # Check if trained model already exists
    if trainer.model_exists(model_path):
        print("Trained model found! Loading existing model and skipping training...")
        trainer.load_model(model_path)
        
        # Evaluate both models
        results = trainer.evaluate_both_models(test_claim_columns=['claim_5'])
        
    else:
        print("No trained model found. Starting training...")
        
        # Train the model using claims 1-4
        trainer.train_model(claim_columns=['claim_1', 'claim_2', 'claim_3', 'claim_4'], epochs=5, batch_size=8)
        
        # Save the model
        trainer.save_model(model_path)
        
        # Evaluate both models
        results = trainer.evaluate_both_models(test_claim_columns=['claim_5'])
    
    print("\nProcess completed!")
    print("="*60)

if __name__ == "__main__":
    main()