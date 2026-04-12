import time
from rag_pipeline import discover

TEST_QUERIES = [
    "chest pain shortness of breath",
    "severe headache blurred vision",
    "type 2 diabetes fatigue weight loss"
]

print("Running MedMatch Evaluation...\n")
latencies = []
similarities = []

for query in TEST_QUERIES:
    start = time.time()
    answer, results, metrics = discover(query, top_k=5)
    latency = round((time.time() - start) * 1000, 1)
    latencies.append(latency)
    sims = [r.get("similarity", 0) for r in results]
    if sims:
        similarities.append(sum(sims)/len(sims))
    print(f"Query: {query[:40]}...")
    print(f"  Latency: {latency}ms | Avg Similarity: {round(sum(sims)/len(sims)*100, 1) if sims else 0}%\n")

print(f"Average Latency:    {round(sum(latencies)/len(latencies), 1)}ms")
print(f"Mean Similarity:    {round(sum(similarities)/len(similarities)*100, 1)}%")
print(f"Total Test Queries: {len(TEST_QUERIES)}")