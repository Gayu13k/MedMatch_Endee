import requests
import os
from dotenv import load_dotenv

load_dotenv()

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
API_KEY = os.getenv("NCBI_API_KEY", "")

def search_pubmed(query: str, max_results: int = 20):
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance"
    }
    if API_KEY:
        params["api_key"] = API_KEY
    r = requests.get(f"{NCBI_BASE}/esearch.fcgi", params=params)
    return r.json()["esearchresult"]["idlist"]

def fetch_abstracts(pmids: list):
    if not pmids:
        return []
    articles = []
    for pmid in pmids:
        try:
            summary_params = {
                "db": "pubmed",
                "id": pmid,
                "retmode": "json"
            }
            if API_KEY:
                summary_params["api_key"] = API_KEY
            s = requests.get(f"{NCBI_BASE}/esummary.fcgi", params=summary_params)
            doc = s.json()["result"][pmid]

            ab_params = {
                "db": "pubmed",
                "id": pmid,
                "retmode": "text",
                "rettype": "abstract"
            }
            if API_KEY:
                ab_params["api_key"] = API_KEY
            ab_r = requests.get(f"{NCBI_BASE}/efetch.fcgi", params=ab_params)
            abstract_text = ab_r.text.strip()

            title = doc.get("title", "")
            if title and abstract_text:
                articles.append({
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract_text,
                    "journal": doc.get("source", ""),
                    "date": doc.get("pubdate", ""),
                    "authors": ", ".join([a["name"] for a in doc.get("authors", [])[:3]]),
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                })
        except Exception as e:
            print(f"Skipping PMID {pmid}: {e}")
            continue
    return articles

def fetch_medical_articles(symptom_query: str, max_results: int = 20):
    print(f"Searching PubMed for: {symptom_query}")
    pmids = search_pubmed(symptom_query, max_results)
    print(f"Found {len(pmids)} articles, fetching abstracts...")
    articles = fetch_abstracts(pmids)
    print(f"Done. Got {len(articles)} articles with abstracts.")
    return articles