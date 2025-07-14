import pandas as pd
import requests
import time
import os
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

def fetch_case_details(api_url):
    """
    Fetch question and conclusion from the API URL
    Returns: dict with 'question' and 'conclusion' keys
    """
    try:
        response = requests.get(api_url)
        
        if response.status_code != 200:
            return {"question": None, "conclusion": None, "url": api_url, "error": f"Status code {response.status_code}"}
        
        data = response.json()
        
        # Extract question and conclusion, clean HTML tags
        question = data.get("question", None)
        conclusion = data.get("conclusion", None)
        
        # Clean HTML tags if present
        if question:
            soup = BeautifulSoup(question, 'html.parser')
            question = soup.get_text(strip=True)
        
        if conclusion:
            soup = BeautifulSoup(conclusion, 'html.parser')
            conclusion = soup.get_text(strip=True)
        
        return {"question": question, "conclusion": conclusion, "url": api_url, "error": None}
        
    except Exception as e:
        return {"question": None, "conclusion": None, "url": api_url, "error": str(e)}

def process_case(row_data):
    """
    Process a single case row
    """
    idx, row = row_data
    href = row['href']
    name = row['name']
    
    details = fetch_case_details(href)
    
    return {
        'idx': idx,
        'name': name,
        'question': details['question'],
        'conclusion': details['conclusion'],
        'error': details['error']
    }

def main():
    # Load the clean_data.csv file
    print("Loading clean_data.csv...")
    df = pd.read_csv('clean_data.csv')
    
    # Output and checkpoint files
    output_file = 'clean_data_with_details.csv'
    checkpoint_file = 'checkpoint_progress.csv'
    
    # Check if there's a checkpoint to resume from
    if os.path.exists(checkpoint_file):
        print("Found checkpoint file, resuming from previous progress...")
        df_checkpoint = pd.read_csv(checkpoint_file)
        # Merge the checkpoint data back into the main dataframe
        df['api_question'] = df_checkpoint['api_question']
        df['api_conclusion'] = df_checkpoint['api_conclusion']
        processed_count = df['api_question'].notna().sum()
        print(f"Resuming: {processed_count} cases already processed")
    else:
        print(f"Found {len(df)} cases to process")
        # Initialize new columns
        df['api_question'] = None
        df['api_conclusion'] = None
        processed_count = 0
    
    # Find cases that still need processing
    unprocessed_mask = df['api_question'].isna()
    unprocessed_indices = df[unprocessed_mask].index.tolist()
    
    if not unprocessed_indices:
        print("All cases already processed!")
        return
    
    print(f"Processing {len(unprocessed_indices)} remaining cases...")
    
    # Prepare data for parallel processing (only unprocessed cases)
    row_data = [(idx, df.loc[idx]) for idx in unprocessed_indices]
    
    # Progress tracking
    completed = processed_count
    total = len(df)
    lock = Lock()
    save_counter = 0
    
    def update_progress():
        nonlocal completed, save_counter
        with lock:
            completed += 1
            save_counter += 1
            if completed % 10 == 0 or completed == total:
                print(f"Completed {completed}/{total} cases ({completed/total*100:.1f}%)")
            
            # Save checkpoint every 50 cases
            if save_counter >= 50:
                print("Saving checkpoint...")
                df.to_csv(checkpoint_file, index=False)
                save_counter = 0
    
    # Process cases in parallel with 10 workers
    print("Starting parallel processing with 10 workers...")
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all tasks
        future_to_case = {executor.submit(process_case, case_data): case_data for case_data in row_data}
        
        # Process completed tasks
        for future in as_completed(future_to_case):
            result = future.result()
            
            # Update the dataframe
            idx = result['idx']
            df.at[idx, 'api_question'] = result['question']
            df.at[idx, 'api_conclusion'] = result['conclusion']
            
            # Log errors
            if result['error']:
                print(f"Error processing {result['name']}: {result['error']}")
            
            update_progress()
    
    # Final save
    print("Saving final results...")
    df.to_csv(output_file, index=False)
    df.to_csv(checkpoint_file, index=False)  # Update checkpoint with final results
    print(f"\nSaved enhanced dataset to {output_file}")
    
    # Print some statistics
    question_count = df['api_question'].notna().sum()
    conclusion_count = df['api_conclusion'].notna().sum()
    
    print(f"\nStatistics:")
    print(f"Cases with questions: {question_count}/{len(df)} ({question_count/len(df)*100:.1f}%)")
    print(f"Cases with conclusions: {conclusion_count}/{len(df)} ({conclusion_count/len(df)*100:.1f}%)")
    
    # Show a sample of the enhanced data
    print(f"\nSample of enhanced data:")
    print(df[['name', 'api_question', 'api_conclusion']].head(3).to_string())
    
    # Clean up checkpoint file when done
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print("Removed checkpoint file (processing complete)")

if __name__ == "__main__":
    main()