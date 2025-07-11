import requests
import pandas as pd
from bs4 import BeautifulSoup

BASE_URL = "https://api.oyez.org/cases"
CASE_WEB_URL = "https://www.oyez.org/cases"
OUT_CSV = "oyez_cases_with_text.csv"

def fetch_all_cases():
    all_cases = []
    page = 0

    while True and page < 5:  # Limit to 1000 pages to avoid infinite loop
        page += 1
        print(f"📄 Fetching page {page}...", end="", flush=True)
        resp = requests.get(BASE_URL, params={"page": page})
        if resp.status_code != 200:
            print(f"\n⛔ Stopped at page {page}. Status code: {resp.status_code}")
            break

        data = resp.json()
        # Handle both list and dict responses
        if isinstance(data, list):
            cases = data
        else:
            cases = data.get("cases", [])
        print(f" ✅ {len(cases)} cases")

        if not cases:
            break

        all_cases.extend(cases)
        # time.sleep(0.2)  # Rate limit

    return all_cases

def get_case_details(case_term, case_docket):
    """
    Returns (has_conclusion: bool, data: dict or None)
    """
    url = f"{CASE_WEB_URL}/{case_term}/{case_docket}"
    try:
        resp = requests.get(url)
        if resp.status_code != 200:
            return False, None
        soup = BeautifulSoup(resp.text, "html.parser")

        # Check for <h2 ng-if="case.conclusion">Conclusion</h2>
        has_conclusion = any(
            h2.get_text(strip=True) == "Conclusion"
            for h2 in soup.find_all("h2", attrs={"ng-if": "case.conclusion", "class": "ng-scope"})
        )
        if not has_conclusion:
            return False, None

        # Extract fields
        def extract_div_text(ng_bind_key):
            tag = soup.find("div", attrs={"ng-bind-html": f"case.{ng_bind_key}"})
            if tag:
                return tag.get_text(separator="\n", strip=True).replace("</p>", "")
            return ""

        return True, {
            "facts": extract_div_text("facts_of_the_case"),
            "question": extract_div_text("question"),
            "conclusion": extract_div_text("conclusion"),
        }

    except Exception as e:
        print(f"⚠️ Error at {url}: {e}")
        return False, {
            "facts": None,
            "question": None,
            "conclusion": None
        }

def extract_fields(case_json):
    """Extract metadata for CSV."""
    term = case_json.get("term")
    docket = case_json.get("docket")
    return {
        "id": case_json.get("id"),
        "term": term,
        "docket_number": docket,
        "name": case_json.get("name"),
        "decision_date": case_json.get("decision_date"),
        "citation": case_json.get("citation"),
        "status": case_json.get("status"),
        "url": f"{CASE_WEB_URL}/{term}/{docket}"
    }

def write_csv(rows, out_path):
    if not rows:
        print("❌ No data to write.")
        return

    # Create DataFrame and save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"\n✅ Done. Wrote {len(rows)} cases to {out_path}")

def main():
    print("🔍 Fetching all Oyez cases...")
    cases = fetch_all_cases()
    print(f"\n📦 Total fetched: {len(cases)}")

    valid_rows = []
    for idx, case in enumerate(cases):
        term = case.get("term")
        docket = case.get("docket")
        print(f"🔎 [{idx+1}/{len(cases)}] {term}/{docket}...", end=" ")

        has_conc, details = get_case_details(term, docket)

        print("✅ Keeping.")
        row = extract_fields(case)
        row["facts"] = details["facts"]
        row["question"] = details["question"]
        row["conclusion"] = details["conclusion"]
        valid_rows.append(row)

    write_csv(valid_rows, OUT_CSV)

if __name__ == "__main__":
    main()
