from endee import Endee, Precision

INDEX_NAME = "medmatch_index"
DIMENSION = 384

client = Endee()

def create_index():
    try:
        client.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            space_type="cosine",
            precision=Precision.INT8
        )
        print(f"Index '{INDEX_NAME}' created.")
    except Exception as e:
        print(f"Index may already exist: {e}")

def upsert_articles(articles: list, embeddings: list):
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
        if len(vectors) == 100:
            index.upsert(vectors)
            vectors = []
    if vectors:
        index.upsert(vectors)
    print(f"Upserted {len(articles)} articles into Endee.")

def search_similar(query_embedding: list, top_k: int = 5):
    index = client.get_index(name=INDEX_NAME)
    return index.query(
        vector=query_embedding,
        top_k=top_k,
        ef=128,
        include_vectors=False
    )