from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from opensearchpy import OpenSearch

# Initialisation FastAPI
app = FastAPI()

# Connexion à OpenSearch
client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    http_compress=True
)

# Chargement du modèle multilingue
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Modèle de données pour la requête
class Query(BaseModel):
    question: str
    lang: str = "fr"  # par défaut français

# Route principale pour poser une question
@app.post("/ask")
def ask(query: Query):
    # Générer le vecteur de la question
    vector = model.encode(query.question).tolist()

    # Requête de similarité avec filtre de langue
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
        return {"answer": hits[0]["_source"]["answer"]}
    else:
        return {"answer": "Désolé, je n’ai pas trouvé de réponse pertinente."}
