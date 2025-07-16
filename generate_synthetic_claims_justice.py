from llm import prompt_vllm
import pandas as pd
import json
from tqdm import tqdm
import os

prompt = """# Instructions:
Carefully generate a truthful factual claim for this court case using natural-sounding language while avoiding highly technical language.
Adhere to the following rules:
- Do not use the case name in the claim.
- Make the claim natural and easy to understand, so have a reasonable length.
 
## Output Format:
Return the claim in a JSON object with the following format:
```json
{{
    "claim": "..."
}}
```
 
## Facts:
{example_sentences}
 
# Question:
{question}

# Conclusion:
{conclusion}
"""

def generate_claims(case, example_sentences, question, conclusion):
    formatted_prompt = prompt.format(
        case=case,
        example_sentences=example_sentences,
        question=question,
        conclusion=conclusion
    )
    
    return prompt_vllm(formatted_prompt)

def process_dataset():
    """Process the entire dataset and generate claims for all Supreme Court cases"""
    
    # Load the dataset
    print("Loading clean_data_with_details.csv...")
    df = pd.read_csv('clean_data_with_details.csv')
    
    # Check if output file already exists to resume processing
    output_file = 'sc_claims.csv'
    if os.path.exists(output_file):
        print("Found existing sc_claims.csv, loading to resume processing...")
        existing_df = pd.read_csv(output_file)
        processed_indices = existing_df.index.tolist()
        print(f"Resuming from {len(processed_indices)} already processed cases")
        df = existing_df.copy()
    else:
        # Add new column for generated claims
        df['generated_claim'] = None
        processed_indices = []
    
    # Find cases that need processing
    unprocessed_mask = df['generated_claim'].isna()
    unprocessed_indices = df[unprocessed_mask].index.tolist()
    
    if not unprocessed_indices:
        print("All cases already processed!")
        return
    
    print(f"Processing {len(unprocessed_indices)} cases...")
    
    # Process each case
    for idx in tqdm(unprocessed_indices, desc="Generating claims"):
        row = df.iloc[idx]
        
        # Skip if missing required data
        if pd.isna(row['api_question']) or pd.isna(row['api_conclusion']) or pd.isna(row['facts']):
            print(f"Skipping case {idx} ({row['name']}) - missing required data")
            df.at[idx, 'generated_claim'] = "MISSING_DATA"
            continue
        
        try:
            # Prepare the case information
            case_name = row['name']
            facts = row['facts']
            question = row['api_question']
            conclusion = row['api_conclusion']
            
            # Create case summary
            case_summary = f"Case: {case_name}\nFacts: {facts}"
            
            # Generate claim using the existing function
            response = generate_claims(
                case=case_summary,
                example_sentences=facts,  # Using facts as example sentences
                question=question,
                conclusion=conclusion
            )
            
            # Try to parse JSON response
            try:
                if response.strip().startswith('{'):
                    claim_data = json.loads(response)
                    claim = claim_data.get('claim', response)
                else:
                    # If not JSON, try to extract claim from response
                    if '"claim"' in response:
                        # Extract claim from JSON-like response
                        start = response.find('"claim"') + len('"claim"')
                        start = response.find('"', start) + 1
                        end = response.find('"', start)
                        claim = response[start:end] if end > start else response
                    else:
                        claim = response.strip()
                
                df.at[idx, 'generated_claim'] = claim
                
            except json.JSONDecodeError:
                # If JSON parsing fails, store the raw response
                df.at[idx, 'generated_claim'] = response.strip()
            
        except Exception as e:
            print(f"Error processing case {idx} ({row['name']}): {str(e)}")
            df.at[idx, 'generated_claim'] = f"ERROR: {str(e)}"
        
        # Save progress every 10 cases
        if (idx + 1) % 10 == 0:
            df.to_csv(output_file, index=False)
    
    # Final save
    print(f"Saving final results to {output_file}...")
    df.to_csv(output_file, index=False)
    
    # Print statistics
    successful_claims = df[df['generated_claim'].notna() & 
                         ~df['generated_claim'].str.startswith('ERROR') & 
                         (df['generated_claim'] != 'MISSING_DATA')].shape[0]
    
    print(f"\nProcessing complete!")
    print(f"Total cases: {len(df)}")
    print(f"Successful claims generated: {successful_claims}")
    print(f"Missing data: {(df['generated_claim'] == 'MISSING_DATA').sum()}")
    print(f"Errors: {df['generated_claim'].str.startswith('ERROR').sum()}")
    
    # Show sample of generated claims
    print(f"\nSample generated claims:")
    sample_claims = df[df['generated_claim'].notna() & 
                      ~df['generated_claim'].str.startswith('ERROR') & 
                      (df['generated_claim'] != 'MISSING_DATA')]['generated_claim'].head(3)
    
    for i, claim in enumerate(sample_claims, 1):
        print(f"{i}. {claim}")

# Run the processing
if __name__ == "__main__":
    process_dataset()