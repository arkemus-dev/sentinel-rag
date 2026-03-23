import fitz  # PyMuPDF
import chromadb
import os
import yaml
import mlflow
from chromadb.utils import embedding_functions
from rich.console import Console

console = Console()

class DocumentIngestor:
    def __init__(self, config_path="../configs/model_config.yaml"):
        # 1. Caricamento Configurazione
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # 2. Configurazione MLflow
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("Sentinel-RAG-Ingestion")
            
        # 3. Inizializzazione Database
        self.client = chromadb.PersistentClient(path=self.config['vector_db']['path'])
        
        # 4. Embedding (BAAI/bge-small-en-v1.5)
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.config['embedding']['model_name'],
            device=self.config['embedding']['device']
        )
        
        self.collection = self.client.get_or_create_collection(
            name=self.config['vector_db']['collection_name'],
            embedding_function=self.ef
        )

    def process_pdf(self, pdf_path, chunk_size=1000, overlap=200):
        if not os.path.exists(pdf_path):
            console.print(f"[bold red]Errore:[/bold red] Il file {pdf_path} non esiste.")
            return

        # Avviamo il tracciamento con MLflow
        run_name = f"Ingest_{os.path.basename(pdf_path)}"
        
        with mlflow.start_run(run_name=run_name):
            try:
                console.print(f"[blue][*][/blue] Elaborazione PDF: [bold]{os.path.basename(pdf_path)}[/bold]...")
                
                # --- Estrazione Testo ---
                doc = fitz.open(pdf_path)
                full_text = "".join([page.get_text() for page in doc])
                doc.close()

                if not full_text.strip():
                    console.print("[yellow]Avviso:[/yellow] Il PDF sembra vuoto o scansionato (senza OCR).")
                    mlflow.log_param("status", "empty_or_no_ocr")
                    return

                # --- Chunking logico ---
                chunks = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size - overlap)]
                
                # --- Logging Parametri su MLflow ---
                mlflow.log_params({
                    "pdf_name": os.path.basename(pdf_path),
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                    "embedding_model": self.config['embedding']['model_name']
                })
                
                # --- Inserimento nel Vector DB ---
                ids = [f"{os.path.basename(pdf_path)}_{i}" for i in range(len(chunks))]
                self.collection.add(
                    documents=chunks,
                    ids=ids,
                    metadatas=[{"source": pdf_path, "chunk": i} for i in range(len(chunks))]
                )
                
                # --- Logging Metriche su MLflow ---
                mlflow.log_metric("total_chunks", len(chunks))
                
                console.print(f"[bold green][+][/bold green] Ingestione riuscita: {len(chunks)} frammenti salvati.")
                console.print(f"[italic cyan][MLflow][/italic cyan] Run registrata con successo.")

            except Exception as e:
                console.print(f"[bold red]Errore critico durante l'ingestione:[/bold red] {e}")
                mlflow.log_param("status", "error")
                mlflow.log_param("error_message", str(e))

if __name__ == "__main__":
    # Assicurati che la cartella dei dati esista
    os.makedirs("./data/raw", exist_ok=True)
    
    ingestor = DocumentIngestor()
    # Eseguiamo il test sul paper
    ingestor.process_pdf("./data/raw/attention_is_all_you_need.pdf")