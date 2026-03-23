import yaml
import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


current_dir = os.path.dirname(os.path.abspath(__file__)) 
root_dir = os.path.dirname(current_dir) # Sale di un livello alla root

prompt_path = os.path.join(root_dir, "configs", "prompt_settings.yaml")

# 1. Caricamento configurazione YAML
if not os.path.exists(prompt_path):
    raise FileNotFoundError(f"⚠️ Errore: Non trovo lo YAML in {prompt_path}")

with open(prompt_path, "r") as f:
    config = yaml.safe_load(f)

# 2. Inizializzazione Embedding (BGE-M3 Multilingua)
model_name = "BAAI/bge-m3"
encode_kwargs = {'normalize_embeddings': True}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name, 
    model_kwargs={'device': 'cuda'}, # Sfruttiamo la tua 4070
    encode_kwargs=encode_kwargs
)

# 3. Setup Retriever (Configurato su K=5 dopo i test di Hit Rate)
vectorstore = Chroma(
    persist_directory="./data/vector_db", 
    embedding_function=embeddings
)
# Usiamo K=5 perché i tuoi test mostrano che l'Hit Rate @3 è solo 0.50
retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) 

# 4. Definizione del Prompt con i tuoi Guardrails
# Integriamo il system_prompt e il rag_template dal tuo file
full_system_instruction = f"{config['system_prompt']}\n\n{config['rag_template']}"

prompt = ChatPromptTemplate.from_messages([
    ("system", full_system_instruction),
    ("human", "{query}")
])

# 5. Inizializzazione LLM (Ollama)
llm = OllamaLLM(model="gpt-oss:20b") # Il tuo modello da 20B

# 6. Costruzione della Chain (LCEL)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "query": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 7. Esecuzione
if __name__ == "__main__":
    query = "Riassumi i punti principali sull'attenzione."
    print(f"\n[*] Sentinel Analyst (LangChain) sta ragionando...\n")
    
    response = rag_chain.invoke(query)
    print(f"RISPOSTA:\n{response}")