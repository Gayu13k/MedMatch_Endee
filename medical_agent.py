from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

AGENT_SYSTEM_PROMPT = """You are MedMatch Agent — an autonomous medical research assistant.

You have access to these tools:
1. search_pubmed(query) — search PubMed for research papers
2. expand_symptoms(symptoms) — break symptoms into medical terms
3. assess_urgency(symptoms) — determine if emergency care is needed

Given a user's symptom description, you must:
STEP 1: assess_urgency — check if this needs emergency care first
STEP 2: expand_symptoms — convert plain English to medical terminology
STEP 3: search_pubmed — search with the expanded medical terms
STEP 4: synthesize — combine all findings into a structured report

Always reason step by step. Format each step as:
THOUGHT: [your reasoning]
ACTION: [tool name]
RESULT: [what you found]

Final answer must include urgency level, expanded terms used, and key findings."""

def run_agent(symptoms: str, search_fn, index_fn):
    """
    Agentic loop — autonomously plans and executes multi-step research.
    """
    steps = []

    # Step 1 — Urgency Assessment
    urgency_prompt = f"""Assess the urgency of these symptoms: {symptoms}
    
    Respond with exactly:
    URGENCY: [EMERGENCY/HIGH/MEDIUM/LOW]
    REASON: [one sentence why]
    SEEK_CARE: [Yes immediately/Yes soon/Monitor/No]"""
    
    urgency_response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": urgency_prompt}
        ],
        max_tokens=150
    )
    urgency_result = urgency_response.choices[0].message.content
    steps.append({"action": "assess_urgency", "result": urgency_result})

    # Step 2 — Expand symptoms to medical terms
    expand_prompt = f"""Convert these symptoms to precise medical terminology for PubMed search: {symptoms}
    
    Respond with exactly:
    MEDICAL_TERMS: [term1, term2, term3]
    PUBMED_QUERY: [optimized search query for PubMed]
    POSSIBLE_CONDITIONS: [condition1, condition2, condition3]"""
    
    expand_response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": expand_prompt}
        ],
        max_tokens=200
    )
    expand_result = expand_response.choices[0].message.content
    steps.append({"action": "expand_symptoms", "result": expand_result})

    # Extract PubMed query from agent output
    pubmed_query = symptoms  # fallback
    for line in expand_result.split('\n'):
        if line.startswith("PUBMED_QUERY:"):
            pubmed_query = line.replace("PUBMED_QUERY:", "").strip()
            break

    # Step 3 — Autonomously index & search with expanded terms
    try:
        count = index_fn(pubmed_query)
        steps.append({"action": "search_pubmed", "result": f"Indexed {count} papers using query: '{pubmed_query}'"})
    except Exception as e:
        steps.append({"action": "search_pubmed", "result": f"Used existing index. Error: {e}"})

    # Step 4 — Search Endee with expanded query
    results = search_fn(symptoms)
    steps.append({"action": "semantic_search", "result": f"Found {len(results)} relevant studies in Endee"})

    return steps, results, urgency_result, expand_result