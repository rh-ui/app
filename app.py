from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from opensearchpy import OpenSearch
from polite import is_not_defined

app = FastAPI()

# Connect to OpenSearch (if you're running FastAPI in Docker, use 'opensearch' instead of 'localhost')
client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    http_compress=True
)

# Load embedding model
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Request model
class Query(BaseModel):
    question: str
    lang: str = "fr"  # default to French

@app.post("/ask")
def ask(query: Query):
    # Create embedding
    vector = model.encode(query.question).tolist()

    # Search OpenSearch
    search_body = {
        "size": 1,
        "query": {
            "bool": {
                "must": [
                    {
                        "knn": {
                            "embedding": {
                                "vector": vector,
                                "k": 1
                            }
                        }
                    },
                    {
                        "term": {
                            "lang": query.lang
                        }
                    }
                ]
            }
        }
    }

    response = client.search(index="faq", body=search_body)
    hits = response["hits"]["hits"]

    if hits:
        answer = hits[0]["_source"]["answer"]
        question = hits[0]["_source"]["question"]
        meta = hits[0]["_source"].get("meta", None)
        score = hits[0]["_score"]

        return {
            "question": question,
            "answer": answer,
            "meta": meta,
            "score": score
        }
    else:
        return {"answer": is_not_defined(query.lang)}
