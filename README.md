# 🛡️ Sentinel-RAG: Advanced Multilingual Retrieval-Augmented Generation

Un framework RAG **Local-First** progettato per l'analisi ad alta precisione di documentazione tecnica. Ottimizzato per l'esecuzione interamente in locale su hardware **NVIDIA RTX 4070 (12GB VRAM)**, garantendo sovranità del dato e latenza minima.

> **🤖 AI Collaboration Note:**
> Questo progetto è stato sviluppato seguendo un workflow di **AI-assisted engineering**. 
> L'intelligenza artificiale generativa (Gemini) è stata utilizzata per:
> * **Code Optimization:** Refactoring della pipeline in **LangChain (LCEL)** e gestione dinamica dei path.
> * **Debugging:** Risoluzione di vulnerabilità critiche (CVE-2025-32434) tramite l'aggiornamento a **Torch v2.6+**.
> * **Theoretical Reinforcement:** Validazione delle metriche di valutazione (**Hit Rate** e **MRR**) per l'ottimizzazione del recupero.

---

## 🎯 Project Goal
L'obiettivo è implementare un **Sentinel Analyst**: un agente specializzato capace di rispondere a quesiti complessi basandosi esclusivamente su un contesto tecnico (PDF), eliminando le allucinazioni tramite **Guardrails** e **System Prompts** strutturati in YAML.

## 🏆 Evaluation & Key Results
A differenza delle implementazioni RAG amatoriali, Sentinel-RAG si basa su una valutazione quantitativa del modulo di retrieval effettuata tramite **BGE-M3**:

* **Hit Rate @3:** **0.50** (Identificato come collo di bottiglia nei test iniziali).
* **Hit Rate @5:** **1.00** (Precisione del recupero del 100% espandendo la finestra di contesto).
* **MRR (Mean Reciprocal Rank) @3:** **0.50**, a indicare che quando il documento corretto è nella Top-3, il modello lo posiziona quasi sempre come primo risultato.
* **Decisione Tecnica:** Il sistema è configurato stabilmente su **K=5** per garantire la massima completezza del contesto fornito all'LLM.

---

## 🛠️ Tech Stack & Methodology

### 1. Semantic Architecture 🌍
* **Embedding Model:** **BGE-M3** (Multilingual, Multi-granularity). Gestisce con precisione termini tecnici in italiano e inglese.
* **Vector Store:** **ChromaDB** (Persistenza locale).
* **LLM:** **GPT-OSS 20B** (via Ollama), selezionato per le elevate capacità di ragionamento autonomo.

### 2. Guardrails & Safety 🛡️
Il comportamento del modello è blindato tramite il file `configs/prompt_settings.yaml`:
* **Contextual Binding:** Divieto di rispondere se l'informazione non è presente nei documenti.
* **Style Guide:** Risposte strutturate in punti elenco, tono tecnico e asciutto.
* **Refusal Logic:** Gestione proattiva di domande fuori tema o malevole.

---

## 📂 Project Structure
```text
├── configs/
│   ├── model_config.yaml    # Parametri hardware e modelli
│   └── prompt_settings.yaml # Definizioni Analyst e Guardrails
├── data/
│   ├── raw/                 # PDF sorgente (es. Attention Is All You Need)
│   └── vector_db/           # Database vettoriale (Escluso da Git)
├── notebooks/
│   └── 01_retrieval_eval.ipynb # Analisi scientifica Hit Rate/MRR
├── src/
│   ├── ingestion.py         # Pipeline di parsing e embedding
│   ├── retrieval.py         # Modulo di ricerca semantica (K=5)
│   └── main.py              # Entry point basato su LangChain LCEL
└── README.md