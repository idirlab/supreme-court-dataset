import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
import time

def process_chunk(chunk_data):
    """Process a chunk of politifact data."""
    chunk_df, case_names = chunk_data
    matching_rows = []
    
    for idx, row in chunk_df.iterrows():
        # Concatenate all cells in the row to create search space
        row_text = ' '.join([str(cell) for cell in row.values if pd.notna(cell)])
        row_text_lower = row_text.lower()
        
        # Check if any case name is present
        has_case_name = False
        matched_case = None
        
        for case_name in case_names:
            if case_name.lower() in row_text_lower:
                has_case_name = True
                matched_case = case_name
                break
        
        # Only add rows that contain case names
        if has_case_name:
            # Add additional columns for tracking matches
            row_dict = row.to_dict()
            row_dict['matched_case_name'] = matched_case
            row_dict['full_text'] = row_text
            matching_rows.append(row_dict)
    
    return matching_rows

def filter_politifact_claims():
    """
    Filter politifact.csv claims based on Supreme Court case relevance.
    
    Criteria:
    - Row text contains any case name from clean_data_with_details.csv
    """
    
    start_time = time.time()
    
    # Read the Supreme Court cases data
    print("Loading Supreme Court cases data...")
    sc_cases = pd.read_csv('/home/a108983/projects/semantic-retrieval/clean_data_with_details.csv')
    
    # Extract case names from the 'name' column
    case_names = sc_cases['name'].dropna().tolist()
    print(f"Found {len(case_names)} Supreme Court case names")
    
    # Read the politifact data
    print("Loading politifact data...")
    politifact_df = pd.read_csv('/home/a108983/projects/semantic-retrieval/politifact.csv')
    print(f"Total politifact records: {len(politifact_df)}")
    
    # Split data into chunks for parallel processing
    num_cores = cpu_count()
    chunk_size = len(politifact_df) // num_cores
    chunks = []
    
    for i in range(0, len(politifact_df), chunk_size):
        chunk = politifact_df.iloc[i:i + chunk_size]
        chunks.append((chunk, case_names))
    
    print(f"Processing with {num_cores} cores using {len(chunks)} chunks...")
    
    # Process chunks in parallel
    with Pool(processes=num_cores) as pool:
        chunk_results = pool.map(process_chunk, chunks)
    
    # Combine results from all chunks
    all_matching_rows = []
    for chunk_result in chunk_results:
        all_matching_rows.extend(chunk_result)
    
    print(f"Processing completed in {time.time() - start_time:.2f} seconds")
    
    # Create DataFrame from matching rows
    if all_matching_rows:
        sc_test_df = pd.DataFrame(all_matching_rows)
        print(f"Total matching rows: {len(sc_test_df)}")
        
        # Save to CSV
        output_file = '/home/a108983/projects/semantic-retrieval/sc_test.csv'
        sc_test_df.to_csv(output_file, index=False)
        print(f"Saved filtered data to {output_file}")
        
        # Print summary statistics
        case_name_matches = sc_test_df['matched_case_name'].apply(lambda x: x != '').sum()
        print(f"Breakdown:")
        print(f"  - Rows with case names: {case_name_matches}")
        print(f"  - Total unique matches: {len(sc_test_df)}")
        
        return sc_test_df
    else:
        print("No matching rows found")
        return None

if __name__ == "__main__":
    result = filter_politifact_claims()