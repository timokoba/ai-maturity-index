import nbformat as nbf

nb = nbf.v4.new_notebook()

nb['cells'] = [
    nbf.v4.new_markdown_cell("""\
# 02 Feature Engineering: Career Pages & ATS Detection
This notebook resolves the official domains for the Fortune 500 US companies, discovers their career pages, and categorizes their Applicant Tracking Systems (ATS).

**Goal**: Prepare a high-quality mapping (`data_clean/fortune500_career_mappings.csv`) for manual review before scraping AI job postings.
"""),
    
    nbf.v4.new_code_cell("""\
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import urllib.parse
from urllib.parse import urljoin, urlparse
from duckduckgo_search import DDGS
from tqdm.auto import tqdm
import time
import socket

# Setup paths
from pathlib import Path
DATA_RAW = Path("../data_raw")
DATA_CLEAN = Path("../data_clean")

# Load Fortune 1000 US and filter to Top 500
df_f1000 = pd.read_csv(DATA_RAW / "fortune_1000_us.csv")
df_f500 = df_f1000[df_f1000['rank'] <= 500].copy().reset_index(drop=True)
print(f"Loaded {len(df_f500)} companies for processing.")
"""),

    nbf.v4.new_markdown_cell("""\
### Step 1: Domain Resolution
Use `duckduckgo_search` to find the official company website by querying `"{company_name} official site"`.
"""),
    nbf.v4.new_code_cell("""\
def get_company_domain(company_name):
    \"\"\"Search for the official domain using DuckDuckGo.\"\"\"
    query = f"{company_name} official site"
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            for res in results:
                url = res.get('href', '')
                # Filter out obvious non-official aggregates
                if any(x in url for x in ['wikipedia.org', 'linkedin.com', 'bloomberg.com', 'forbes.com', 'finance.yahoo', 'sec.gov']):
                    continue
                
                # Extract clean domain
                parsed_uri = urlparse(url)
                domain = f"{parsed_uri.scheme}://{parsed_uri.netloc}"
                return domain
    except Exception as e:
        pass
    return None
"""),

    nbf.v4.new_markdown_cell("""\
### Step 2: Career URL Extraction
Attempt standard career paths first (`/careers`, `/jobs`). If they fail (404), fetch the homepage and extract all `<a>` tags matching career-related keywords.
"""),
    nbf.v4.new_code_cell("""\
PROBE_PATHS = ['/careers', '/jobs', '/about/careers', '/work-with-us', '/join-us']
CAREER_KEYWORDS = ['career', 'jobs', 'join', 'work', 'employment']

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

def find_career_url(domain):
    \"\"\"Find the career page URL for a given domain.\"\"\"
    if not domain:
        return None, "low"
    
    # 1. Probing common paths
    for path in PROBE_PATHS:
        test_url = urljoin(domain, path)
        try:
            r = requests.head(test_url, headers=HEADERS, timeout=5, allow_redirects=True)
            if r.status_code == 200:
                # Basic check: do they actually have HTML content?
                r_get = requests.get(test_url, headers=HEADERS, timeout=5)
                if 'job' in r_get.text.lower() or 'career' in r_get.text.lower():
                    return test_url, "high"
        except requests.RequestException:
            continue
            
    # 2. Homepage extraction
    try:
        r = requests.get(domain, headers=HEADERS, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')
        
        links = soup.find_all('a', href=True)
        for link in links:
            href = link['href']
            text = link.get_text().strip().lower()
            
            # Match text or URL against keywords
            if any(kw in text for kw in CAREER_KEYWORDS) or any(kw in href.lower() for kw in CAREER_KEYWORDS):
                # Unlikely to be an investor relations page
                if 'investor' not in href.lower() and 'investor' not in text:
                    full_url = urljoin(domain, href)
                    return full_url, "medium"
                    
    except Exception as e:
        return None, "low"
        
    return domain, "low" # Fallback to domain if nothing found
"""),

    nbf.v4.new_markdown_cell("""\
### Step 3: ATS Detection
Parse the final career URL (and lightly fetch its content to trace redirects) to classify the ATS platform.
"""),
    nbf.v4.new_code_cell("""\
# Specific signatures that identify ATS platforms
ATS_SIGNATURES = {
    'Greenhouse': ['greenhouse.io'],
    'Workday': ['myworkdayjobs.com'],
    'Lever': ['lever.co'],
    'Taleo': ['taleo.net'],
    'SAP SuccessFactors': ['successfactors.com', 'successfactors.eu'],
    'iCIMS': ['icims.com'],
    'SmartRecruiters': ['smartrecruiters.com'],
    'Eightfold AI': ['eightfold.ai']
}

def detect_ats(career_url):
    \"\"\"Detect ATS type based on URL and redirect tracing.\"\"\"
    if not career_url:
        return "custom"
        
    # Check URL directly first
    for ats, signatures in ATS_SIGNATURES.items():
        if any(sig in career_url.lower() for sig in signatures):
            return ats
            
    # Tracing redirects and simple HTML match
    try:
        r = requests.get(career_url, headers=HEADERS, timeout=10, allow_redirects=True)
        final_url = r.url.lower()
        html_text = r.text.lower()
        
        for ats, signatures in ATS_SIGNATURES.items():
             if any(sig in final_url for sig in signatures) or any(sig in html_text for sig in signatures):
                return ats
    except requests.RequestException:
        pass
        
    return "custom"
"""),

    nbf.v4.new_markdown_cell("""\
### Step 4: Pipeline Execution
Run the logic systematically across the Fortune 500 US dataset. Note: Processing 500 domains is deliberately sequential with error handling to avoid Search API bans and handle poorly configured corporate firewalls.
"""),
    nbf.v4.new_code_cell("""\
results = []

# Using a subset (e.g. 5) for fast testing. To run all, change to df_f500.iterrows()
print("Starting pipeline. Note: processing completely takes ~10-15 minutes...")
for idx, row in tqdm(df_f500.iterrows(), total=len(df_f500)):
    company = row['company']
    
    # Random sleep to prevent rate limiting from DDG
    time.sleep(1) 
    
    domain = get_company_domain(company)
    career_url, confidence = find_career_url(domain)
    ats_type = detect_ats(career_url)
    
    results.append({
        'company_name': company,
        'domain': domain,
        'career_url': career_url,
        'ats_type': ats_type,
        'confidence_score': confidence
    })
    
df_results = pd.DataFrame(results)

output_path = DATA_CLEAN / "fortune500_career_mappings.csv"
df_results.to_csv(output_path, index=False)
print(f"\\nPipeline complete. Results saved to: {output_path}")

# Display Sample
df_results.head(10)
"""),
    nbf.v4.new_markdown_cell("""\
### Step 5: High-Level Audit
Check how many jobs were automatically classified into standardized ATS systems vs custom ones, and how many are missing.
"""),
    nbf.v4.new_code_cell("""\
print("ATS SYSTEM DISTRIBUTION:")
print(df_results['ats_type'].value_counts())

print("\\nCONFIDENCE SCORE DISTRIBUTION:")
print(df_results['confidence_score'].value_counts())

print("\\nMISSING URLS (Requires Manual Fix):")
print(len(df_results[df_results['career_url'].isna()]))
""")
]

with open(r'd:\A Studium\MSc Management and Technology\(5) Wintersemester 2025 26\Masterarbeit\Code\notebooks\02_feature_engineering.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
