import ollama
import yaml
from retrieval import SentinelRetriever
from rich.console import Console
from rich.markdown import Markdown

console = Console()

class SentinelEngine:
    def __init__(self, config_path="configs/model_config.yaml", prompt_path="configs/prompt_settings.yaml"):

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        with open(prompt_path, 'r') as f:
            self.prompts = yaml.safe_load(f)
        
        print(f"Chiavi trovate nello YAML: {self.prompts.keys()}")
        self.retriever = SentinelRetriever(config_path)

    def ask(self, query):
        # 1. Recupero contesto
        context = self.retriever.get_relevant_chunks(query)
        
        # 2. Costruzione Prompt
        system_msg = self.prompts['system_prompt']
        user_msg = f"CONTESTO:\n{context}\n\nQUESITO: {query}"
        
        try:
            console.print("[italic blue]Sentinel sta ragionando...[/italic blue]")
            response = ollama.generate(
                model=self.config['llm']['model_name'],
                prompt=f"{system_msg}\n\n{user_msg}",
                options={"temperature": self.config['llm']['temperature']}
            )
            return response['response']
        except Exception as e:
            return f"Errore nella generazione (Ollama è attivo?): {e}"

def main():
    engine = SentinelEngine()
    console.print("[bold cyan]Benvenuto in Sentinel-RAG v1.0[/bold cyan]")
    
    while True:
        query = console.input("\n[bold yellow]Domanda (o 'exit'):[/bold yellow] ")
        if query.lower() == 'exit': break
        
        answer = engine.ask(query)
        console.print("\n[bold green]RISPOSTA:[/bold green]")
        console.print(Markdown(answer))

if __name__ == "__main__":
    main()