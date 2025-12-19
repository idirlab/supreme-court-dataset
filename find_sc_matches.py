import pandas as pd
import re
import os
import multiprocessing
import requests
from bs4 import BeautifulSoup
import time

# Global variable for workers
sc_cases_global = None

def init_worker(cases):
    global sc_cases_global
    sc_cases_global = cases

def scrape_and_search(row_data):
    idx, url, claim, verdict = row_data
    matches = []
    
    if not isinstance(url, str) or not url.startswith('http'):
        return matches

    try:
        # Add a user-agent to be polite/avoid blocking
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; ResearchBot/1.0)'}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return matches
            
        soup = BeautifulSoup(response.content, 'html.parser')
        # Get text from the article body. 
        text = soup.get_text(" ", strip=True)
        text_lower = text.lower()
        
        # Search for cases
        for case in sc_cases_global:
            # Case insensitive search
            if case.lower() in text_lower:
                matches.append({
                    'Politifact_ID': idx,
                    'Claim': claim,
                    'Verdict': verdict,
                    'SC_Case_Match': case,
                    'Factcheck_Url': url
                })
            
    except Exception as e:
        # Silently fail on scrape errors to keep output clean
        pass
        
    return matches

def main():
    base_path = '/home/a108983/projects/supreme-court-dataset'
    sc_file = os.path.join(base_path, 'clean_data_with_details.csv')
    pf_file = os.path.join(base_path, 'data_store/politifact.csv')

    print("Loading datasets...")
    try:
        sc_df = pd.read_csv(sc_file)
        # Filter for valid names
        sc_cases = sc_df['name'].dropna().unique()
        # Filter out very short names that might be false positives (e.g. "In re A")
        # Keep if it has "v." or length > 4
        filtered_cases = [c for c in sc_cases if len(c) > 4]
        print(f"Loaded {len(sc_cases)} cases. Using {len(filtered_cases)} for search.")
    except Exception as e:
        print(f"Error loading SC data: {e}")
        return

    try:
        # on_bad_lines='skip' to handle potential CSV formatting issues
        pf_df = pd.read_csv(pf_file, on_bad_lines='skip', engine='python')
        print(f"Loaded {len(pf_df)} Politifact items.")
    except Exception as e:
        print(f"Error loading Politifact data: {e}")
        return

    # Combine text for initial filtering
    pf_df['search_text'] = pf_df['Claim'].fillna('') + " " + \
                           pf_df['Review'].fillna('') + " " + \
                           pf_df['Review Summary'].fillna('')
    
    # Pre-filter for efficiency - only scrape pages that mention Supreme Court keywords in metadata
    print("Filtering Politifact data for potential SC relevance (metadata check)...")
    keywords = ['Supreme Court', 'SCOTUS', 'High Court', 'v.', 'vs.']
    # Create a regex pattern for keywords
    pattern = '|'.join([re.escape(k) for k in keywords])
    
    mask = pf_df['search_text'].str.contains(pattern, case=False, regex=True)
    relevant_pf = pf_df[mask].copy()
    print(f"Found {len(relevant_pf)} potentially relevant items to scrape.")

    if len(relevant_pf) == 0:
        print("No relevant items found.")
        return

    print(f"Scraping and searching using {multiprocessing.cpu_count()} processes...")
    
    # Prepare tasks
    tasks = []
    for idx, row in relevant_pf.iterrows():
        tasks.append((idx, row['Factcheck Url'], row['Claim'], row['Verdict']))
    
    all_results = []
    
    # Use a pool to process URLs in parallel
    # We pass the list of cases to each worker once via initializer
    with multiprocessing.Pool(processes=multiprocessing.cpu_count(), initializer=init_worker, initargs=(filtered_cases,)) as pool:
        # Use imap_unordered for potentially better responsiveness if we were showing progress
        # But map is fine.
        results_nested = pool.map(scrape_and_search, tasks)
        
    for res in results_nested:
        all_results.extend(res)

    print(f"Found {len(all_results)} matches.")
    
    if all_results:
        results_df = pd.DataFrame(all_results)
        output_file = os.path.join(base_path, 'politifact_sc_matches_scraped.csv')
        results_df.to_csv(output_file, index=False)
        print(f"Saved matches to {output_file}")
        
        # Print sample
        print("\nSample Matches:")
        print(results_df[['SC_Case_Match', 'Claim']].head(10).to_string())

if __name__ == "__main__":
    main()
