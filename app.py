from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from opensearchpy import OpenSearch
from polite import is_not_defined
from deepseek_client import DeepSeekClient

app = FastAPI()

# Connect to OpenSearch
client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    http_compress=True
)

# Load embedding model
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Initialize DeepSeek client
deepseek_client = DeepSeekClient()

# Request model
class Query(BaseModel):
    question: str
    lang: str = "fr"  # default to French

@app.post("/ask")
async def ask(query: Query):
    # Create embedding
    vector = model.encode(query.question).tolist()
    
    # Search OpenSearch for top 5 matches
    search_body = {
        "size": 5,  # Get top 5 matches instead of 1
        "query": {
            "bool": {
                "must": [
                    {
                        "knn": {
                            "embedding": {
                                "vector": vector,
                                "k": 5  # Get top 5 matches
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
        # Prepare FAQ matches for DeepSeek
        faq_matches = []
        for hit in hits:
            match_data = {
                "question": hit["_source"]["question"],
                "answer": hit["_source"]["answer"],
                "meta": hit["_source"].get("meta", None),
                "score": hit["_score"]
            }
            faq_matches.append(match_data)
        
        # Get response from DeepSeek API
        deepseek_response = await deepseek_client.get_response(query.question, faq_matches)
        
        return {
            "question": query.question,
            "answer": deepseek_response,
            "faq_matches": faq_matches,  # Include original matches for reference
            "source": "deepseek"
        }
    else:
        # No matches found, return original behavior
        return {"answer": is_not_defined(query.lang)}
    