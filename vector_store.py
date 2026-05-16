from endee import Endee, Precision
import os

INDEX_NAME = "medmatch_index"
DIMENSION = 384

try:
    client = Endee()
    ENDEE_AVAILABLE = True
except Exception:
    ENDEE_AVAILABLE = False
    client = None

def create_index():
    if not ENDEE_AVAILABLE:
        return
    try:
        client.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            space_type="cosine",
            precision=Precision.INT8
        )
    except Exception as e:
        print(f"Index may already exist: {e}")

def upsert_articles(articles, embeddings):
    if not ENDEE_AVAILABLE:
        return
    index = client.get_index(name=INDEX_NAME)
    vectors = []
    for article, embedding in zip(articles, embeddings):
        vectors.append({
            "id": article["pmid"],
            "vector": embedding,
            "meta": {
                "title": article["title"],
                "journal": article["journal"],
                "date": article["date"],
                "authors": article["authors"],
                "url": article["url"],
                "abstract_snippet": article["abstract"][:500]
            },
            "filter": {
                "journal": article["journal"][:48] if article["journal"] else "unknown"
            }
        })
    if vectors:
        index.upsert(vectors)

def search_similar(query_embedding, top_k=5):
    if not ENDEE_AVAILABLE:
        return []
    index = client.get_index(name=INDEX_NAME)
    return index.query(
        vector=query_embedding,
        top_k=top_k,
        ef=128,
        include_vectors=False
    )

def is_endee_available():
    return ENDEE_AVAILABLE