from sentence_transformers import SentenceTransformer
from groq import Groq
from vector_store import create_index, upsert_articles, search_similar
from pubmed_fetcher import fetch_medical_articles
import os
from dotenv import load_dotenv

load_dotenv()

embedder = SentenceTransformer("all-MiniLM-L6-v2")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def index_symptoms(symptom_query: str):
    articles = fetch_medical_articles(symptom_query, max_results=20)
    if not articles:
        return 0
    texts = [f"{a['title']}. {a['abstract'][:800]}" for a in articles]
    embeddings = embedder.encode(texts, show_progress_bar=True).tolist()
    create_index()
    upsert_articles(articles, embeddings)
    return len(articles)
import time

def discover(user_query: str, top_k: int = 5):
    start_time = time.time()

    query_embedding = embedder.encode([user_query])[0].tolist()
    results = search_similar(query_embedding, top_k=top_k)

    latency_ms = round((time.time() - start_time) * 1000, 1)

    if not results:
        return "No relevant research found.", [], {}

    # Build context
    context_parts = []
    for r in results:
        meta = r.get("meta", {})
        context_parts.append(
            f"Title: {meta.get('title', 'N/A')}\n"
            f"Journal: {meta.get('journal', 'N/A')} ({meta.get('date', '')})\n"
            f"Abstract: {meta.get('abstract_snippet', '')}\n"
        )
    context = "\n---\n".join(context_parts)

    prompt = f"""You are a medical research assistant. Based ONLY on the research abstracts below, provide:

1. **Possible Conditions**: List 2-3 conditions the research discusses (as comma-separated tags)
2. **Key Findings**: What treatments or findings are mentioned
3. **Severity Indicators**: Any urgency or warning signs mentioned
4. **Caveats**: Important limitations

Format your response exactly like this:
CONDITIONS: [condition1, condition2, condition3]
FINDINGS: [your findings here]
SEVERITY: [Low/Medium/High] - [reason]
CAVEATS: [caveats here]

Always end with: "Please consult a qualified healthcare professional."

User symptoms: {user_query}

Research abstracts:
{context}

Summary:"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600
    )
    answer = response.choices[0].message.content

    # Compute evaluation metrics
    similarities = [r.get("similarity", 0) for r in results]
    avg_similarity = round(sum(similarities) / len(similarities) * 100, 1)
    precision_threshold = 0.5  # consider "relevant" if similarity > 0.5
    relevant = [s for s in similarities if s > precision_threshold]
    precision_at_3 = round(len([s for s in similarities[:3] if s > precision_threshold]) / 3 * 100, 1)
    precision_at_5 = round(len([s for s in similarities[:5] if s > precision_threshold]) / min(5, len(similarities)) * 100, 1)

    metrics = {
        "latency_ms": latency_ms,
        "avg_similarity": avg_similarity,
        "precision_at_3": precision_at_3,
        "precision_at_5": precision_at_5,
        "total_results": len(results)
    }

    return answer, results, metrics




