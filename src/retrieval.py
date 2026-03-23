import chromadb
import yaml
from chromadb.utils import embedding_functions

class SentinelRetriever:
    def __init__(self, config_path="configs/model_config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        self.client = chromadb.PersistentClient(path=config['vector_db']['path'])
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config['embedding']['model_name']
        )
        self.collection = self.client.get_collection(
            name=config['vector_db']['collection_name'],
            embedding_function=self.ef
        )

    def get_relevant_chunks(self, query, n_results=3):
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            # Ritorna i documenti uniti come stringa di contesto
            return "\n---\n".join(results['documents'][0])
        except Exception as e:
            return f"Errore nel recupero dati: {e}"