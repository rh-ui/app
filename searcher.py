from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer

client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}]
)

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

INDEX_NAME = "faq"

def hybrid_search(user_question, k=3, score_threshold=0.5):
    embedding = model.encode(user_question).tolist()

    query_body = {
        "size": k,
        "query": {
            "bool": {
                "should": [
                    {
                        "knn": {
                            "embedding": {
                                "vector": embedding,
                                "k": k
                            }
                        }
                    },
                    {
                        "match": {
                            "question": {
                                "query": user_question,
                                "fuzziness": "AUTO"
                            }
                        }
                    }
                ]
            }
        }
    }

    response = client.search(index=INDEX_NAME, body=query_body)
    hits = response["hits"]["hits"]

    # Filter hits by score threshold (on kNN score or text match)
    filtered = []
    for hit in hits:
        score = hit.get("_score", 0)
        if score >= score_threshold:
            filtered.append(hit)

    return filtered

if __name__ == "__main__":
    question = input("Ask your question: ")
    results = hybrid_search(question)

    if not results:
        print("Sorry, no good answer found.")
    else:
        for hit in results:
            source = hit["_source"]
            print(f"Q: {source['question']}")
            print(f"A: {source['answer']}")
            if "meta" in source:
                print(f"More info: {source['meta']}")
            print(f"Score: {hit['_score']:.4f}\n")
