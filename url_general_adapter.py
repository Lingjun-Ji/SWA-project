# This cell wraps the core functionality of Url General.ipynb to be compatible with SWA_UI.py
# It replaces the original urlscrapper.generate_urls(user_inputs, matched_orgs)

def generate_urls(user_inputs: dict, matched_orgs: list):
    import pandas as pd
    import requests
    import os
    import re
    from urllib.parse import urlparse, urljoin
    from datetime import datetime, timedelta

    API_KEY ="API_KEY"
    CX = 'CX'
    MAX_RESULTS = 10

    def google_search(query, start_year, max_results=10):
        results = []
        year_keywords = " OR ".join(str(y) for y in range(start_year, datetime.now().year + 1))
        query = f"{query} {year_keywords}"
        for start in range(1, max_results, 10):
            url = f'https://www.googleapis.com/customsearch/v1?key={API_KEY}&cx={CX}&q={query}&start={start}'
            response = requests.get(url).json()
            print("🧪 Google API Raw Response:", response)
            items = response.get('items', [])
            results.extend([item['link'] for item in items])
            if len(items) < 10:
                break
        return results

    def detect_file_type(url):
        path = urlparse(url).path.lower()
        if path.endswith(".pdf"):
            return "PDF"
        elif path.endswith(".xls") or path.endswith(".xlsx"):
            return "Excel"
        elif path.endswith(".html") or path.endswith(".htm") or not "." in path:
            return "HTML"
        return "Other"

    def is_trusted_link(link, org_name):
        netloc = urlparse(link).netloc.lower()

        # clean organization name
        base_name = re.sub(r'\s+|\([^)]*\)', '', org_name.lower())
        if base_name and base_name in netloc:
            return True
        
        abbrev_match = re.search(r'\(([^)]+)\)', org_name)
        if abbrev_match and abbrev_match.group(1).lower() in netloc:
            return True
        
        return False

    # Extract fields from user_inputs
    start_year = int(user_inputs["year"])
    frequency_months = int(user_inputs.get("Frequency", "1"))
    frequency_days = frequency_months * 30
    SDG_Goals = ", ".join(user_inputs.get("sdg_labels", []))
    country = ", ".join(user_inputs.get("country", []))
    #industry_input = ", ".join(user_inputs.get("industry", []))
    doc_types = [d.split()[0] for d in user_inputs.get("doc_labels", [])]

    rows = []
    current_date = datetime.now()
    seen_links = {}  # key: (org, url), value: last_scraped datetime

    for org in matched_orgs:
        org_name = org["organisation_name"]
        industry = org.get("industry", "")
        division = org.get("division", "")

        for doc_type in doc_types:
            if doc_type.lower() == "pdf":
                #filetype:pdf
                query = f"{org_name} sustainability OR ESG filetype:pdf Annual Report"
            else:
                # non-PDF keep keywords
                query = f"{org_name} sustainability OR ESG Australia OR Annual Report"
            print(f"🔎 Searching for: {query} from {start_year} to {datetime.now().year}")
            links = google_search(query, start_year)

            for link in links:
                key = (org_name, link)
                if key in seen_links:
                    last_scraped = seen_links[key]
                    if (current_date - last_scraped).days < frequency_days:
                        print(f"⏩ Skipped (duplicate within frequency window): {link}")
                        continue

                file_type = detect_file_type(link)
                trusted = is_trusted_link(link, org_name)

                print(f"Link: {link}")
                print(f"Detected File Type: {file_type} | Expected: one of {doc_types}")

                if file_type.lower() != doc_type.lower():
                    continue

                rows.append({
                    "Organization": org_name,
                    "Division": division,
                    "Industry": industry,
                    "Country": country,
                    "SDG_Goals": SDG_Goals if SDG_Goals else "None selected",
                    "Year Range Start": start_year,
                    "URL": link,
                    "File Type": file_type,
                    "Flag": "Trusted" if trusted else "Third-party",
                    "Last Scraped": current_date.strftime("%Y-%m-%d")
                })

                seen_links[key] = current_date

    df = pd.DataFrame(rows)
    output_path = os.path.abspath("generated_urls.csv")
    df.to_csv(output_path, index=False)
    print("✅ URL results saved to:", output_path)
