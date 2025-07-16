import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import time
from urllib.parse import urljoin, urlparse
import warnings
warnings.filterwarnings('ignore')

import re
from typing import Optional
import concurrent.futures
from threading import Lock

import requests
from bs4 import BeautifulSoup
from ezmm import MultimodalSequence
from markdownify import MarkdownConverter


def md(soup, **kwargs):
    """Converts a BeautifulSoup object into Markdown."""
    return MarkdownConverter(**kwargs).convert_soup(soup)


def postprocess_scraped(text: str) -> str:
    # Remove any excess whitespaces
    text = re.sub(r' {2,}', ' ', text)

    # remove any excess newlines
    text = re.sub(r'(\n *){3,}', '\n\n', text)

    return text


def scrape_url_content(url: str) -> Optional[MultimodalSequence]:
    """Fallback scraping script with a 15-second timeout."""
    headers = {
        'User-Agent': 'Mozilla/4.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    }
    def _scrape():
        try:
            page = requests.get(url, headers=headers, timeout=5)
            # Handle any request errors
            if page.status_code == 403:
                return None
            elif page.status_code == 404:
                return None
            page.raise_for_status()
            soup = BeautifulSoup(page.content, 'html.parser')
            if soup.article:
                soup = soup.article
            text = md(soup)
            text = postprocess_scraped(text)
            return text
        except requests.exceptions.RequestException:
            return None
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_scrape)
        try:
            result = future.result(timeout=15)
            if result is None:
                return None
            return result
        except concurrent.futures.TimeoutError:
            return "Unable to Scrape"


def process_row(row_data):
    """Process a single row of politifact data"""
    idx, row, case_variations = row_data
    
    factcheck_url = row.get('Factcheck Url', '')
    
    if pd.isna(factcheck_url) or factcheck_url == '':
        return None
    
    print(f"  Scraping URL {idx}: {factcheck_url}")
    
    # Scrape the content
    content = scrape_url_content(factcheck_url)
    
    if content:
        content_lower = content.lower()
        
        # Check if "supreme court" is in the content
        has_supreme_court = "supreme court" in content_lower
        supreme_court_context = ""
        
        if has_supreme_court:
            # Find the position of "supreme court" and extract context
            pos = content_lower.find("supreme court")
            start = max(0, pos - 50)
            end = min(len(content), pos + len("supreme court") + 50)
            supreme_court_context = content[start:end]
        
        # Check if any case name is in the content
        has_case_name = False
        case_name_context = ""
        found_case_name = ""
        
        for case_name in case_variations:
            if case_name in content_lower:
                has_case_name = True
                found_case_name = case_name
                # Find the position of case name and extract context
                pos = content_lower.find(case_name)
                start = max(0, pos - 50)
                end = min(len(content), pos + len(case_name) + 50)
                case_name_context = content[start:end]
                break
        
        if has_supreme_court or has_case_name:
            print(f"    ✓ Found relevant content!")
            
            if has_supreme_court:
                print(f"      Supreme Court context: ...{supreme_court_context}...")
            
            if has_case_name:
                print(f"      Case name '{found_case_name}' context: ...{case_name_context}...")
            
            row_dict = row.to_dict()
            row_dict['scraped_content'] = content
            row_dict['has_supreme_court'] = has_supreme_court
            row_dict['has_case_name'] = has_case_name
            row_dict['supreme_court_context'] = supreme_court_context
            row_dict['case_name_context'] = case_name_context
            row_dict['found_case_name'] = found_case_name
            return row_dict
        else:
            print(f"    ✗ Not relevant")
    else:
        print(f"    ✗ Failed to scrape content")
    
    return None


def main():
    print("Loading Supreme Court case data...")
    try:
        sc_data = pd.read_csv('clean_data_with_details.csv')
        print(f"Loaded {len(sc_data)} Supreme Court cases")
        
        # Extract case names and clean them
        case_names = sc_data['name'].tolist()
        
        # Create variations of case names for better matching
        case_variations = set()
        for name in case_names:
            if pd.notna(name):
                case_variations.add(name.lower())
                case_variations.add(name)
        
        print(f"Created {len(case_variations)} case name variations")
        
    except Exception as e:
        print(f"Error loading Supreme Court data: {e}")
        return

    print("\nLoading Politifact data...")
    try:
        # Load politifact data in chunks to handle large file
        chunk_size = 1000
        relevant_rows = []
        save_counter = 0
        
        # Try to load existing results if they exist
        try:
            existing_df = pd.read_csv('sc_test.csv')
            relevant_rows = existing_df.to_dict('records')
            print(f"Loaded {len(relevant_rows)} existing results from sc_test.csv")
            save_counter = len(relevant_rows)
        except FileNotFoundError:
            print("No existing results found, starting fresh")
        
        for chunk_num, chunk in enumerate(pd.read_csv('politifact.csv', chunksize=chunk_size, low_memory=False)):
            print(f"Processing chunk {chunk_num + 1}...")
            
            # Prepare data for parallel processing
            row_data_list = []
            for idx, row in chunk.iterrows():
                row_data_list.append((idx, row, case_variations))
            
            # Process rows in parallel with 10 workers
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                # Submit all tasks
                futures = [executor.submit(process_row, row_data) for row_data in row_data_list]
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        if result is not None:
                            relevant_rows.append(result)
                            save_counter += 1
                            
                            # Save every 100 claims
                            if save_counter % 100 == 0:
                                temp_df = pd.DataFrame(relevant_rows)
                                temp_df.to_csv('sc_test.csv', index=False)
                                print(f"    💾 Saved {save_counter} results to sc_test.csv")
                                
                    except Exception as e:
                        print(f"Error processing row: {e}")
            
            print(f"  Found {len(relevant_rows)} total relevant rows so far")
            
            # Optional: add a small delay between chunks to be respectful
            time.sleep(0.5)
        
        print(f"\nFound {len(relevant_rows)} relevant rows")
        
        if relevant_rows:
            # Final save
            result_df = pd.DataFrame(relevant_rows)
            result_df.to_csv('sc_test.csv', index=False)
            print(f"Final save: {len(result_df)} relevant rows to sc_test.csv")
            
            # Show summary
            print("\nSummary:")
            print(f"Rows with 'supreme court' mention: {result_df['has_supreme_court'].sum()}")
            print(f"Rows with case name mention: {result_df['has_case_name'].sum()}")
            print(f"Total relevant rows: {len(result_df)}")
            
            # Show first few claims
            print("\nFirst few relevant claims:")
            for i, row in result_df.head(3).iterrows():
                print(f"{i+1}. {row['Claim'][:100]}...")
                print(f"   URL: {row['Factcheck Url']}")
                print(f"   Supreme Court: {row['has_supreme_court']}, Case Name: {row['has_case_name']}")
                
                if row['has_supreme_court'] and row['supreme_court_context']:
                    print(f"   Supreme Court context: ...{row['supreme_court_context']}...")
                
                if row['has_case_name'] and row['case_name_context']:
                    print(f"   Case name '{row['found_case_name']}' context: ...{row['case_name_context']}...")
                
                print()
        else:
            print("No relevant rows found")
            
    except Exception as e:
        print(f"Error processing Politifact data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()