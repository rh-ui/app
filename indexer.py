import json
import uuid
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer

# Connexion à OpenSearch
client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    http_compress=True
)

# Charger le modèle multilingue
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Créer l'index s'il n'existe pas
INDEX_NAME = "faq"
VECTOR_DIM = 384

def create_index():
    if not client.indices.exists(index=INDEX_NAME):
        client.indices.create(
            index=INDEX_NAME,
            body={
                "settings": {
                    "index": {
                        "knn": True
                    }
                },
                "mappings": {
                    "properties": {
                        "question": {"type": "text"},
                        "answer": {"type": "text"},
                        "lang": {"type": "keyword"},
                        "embedding": {
                            "type": "knn_vector",
                            "dimension": VECTOR_DIM
                        }
                    }
                }
            }
        )
        print("✅ Index 'faq' créé.")
    else:
        print("ℹ️ Index 'faq' existe déjà.")

def index_faq_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    total = 0
    for entry in dataset:
        questions = entry["question"]
        reponses = entry["reponse"]

        for lang in questions:
            for question in questions[lang]:
                if lang not in reponses:
                    continue
                answer = reponses[lang][0]
                embedding = model.encode(question).tolist()
                doc = {
                    "question": question,
                    "answer": answer,
                    "lang": lang,
                    "embedding": embedding
                }
                client.index(index=INDEX_NAME, id=str(uuid.uuid4()), body=doc)
                total += 1

    print(f"✅ {total} questions indexées dans OpenSearch.")

if __name__ == "__main__":
    create_index()
    index_faq_data("contact_final.json")
