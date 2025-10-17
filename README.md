# 📘 Knowledge-base Search Engine — Documentation

## 🔗 Quick Links
- **DEMO VIDEO FILE:** [Click Here](https://drive.google.com/file/d/1_Pqf8hIO8-o3JCM4TpzvDWDT8XyzaH46/view?usp=sharing)  
- **GITHUB CODE:** [Click Here](https://github.com/karthiksagarN/knowledge_base_search_engine)

## 🔍 Overview
The **Knowledge-base Search Engine** is an AI-powered system that allows users to query across multiple documents (PDFs) and receive synthesized, context-aware answers.  
It leverages **Retrieval-Augmented Generation (RAG)** using **LangChain**, **ChromaDB**, and **Ollama LLMs** to retrieve relevant information and generate concise responses.

---

## 🎯 Objective
> Search across uploaded documents and generate synthesized answers using Large Language Models (LLMs) integrated with retrieval-based context.

---

## 🧠 Core Features
- 📂 **Document Upload** — Add multiple PDFs to the knowledge base.  
- 🧮 **Embeddings** — Documents are chunked and converted into vector embeddings using `OllamaEmbeddings` (`nomic-embed-text` model).  
- 🔎 **Retrieval** — Similarity-based search retrieves the most relevant chunks from ChromaDB.  
- 💬 **Query Answering** — Answers are generated via `OllamaLLM` (Mistral model).  
- 📊 **Interactive UI** — Built with Streamlit for document management, querying, and chat history.  
- ⚙️ **Database Control** — Supports reset, viewing document stats, and incremental updates.

---

## 🏗️ System Architecture
```
                 ┌──────────────────────┐
                 │   User Interface     │
                 │ (Streamlit Web App)  │
                 └──────────┬───────────┘
                            │
                            ▼
                  ┌────────────────────┐
                  │ Query Processing   │
                  │ (LangChain + LLM)  │
                  └────────────────────┘
                            │
                            ▼
                  ┌────────────────────┐
                  │  Vector Database   │
                  │ (Pinecone)         │
                  └────────────────────┘
                            │
                            ▼
                  ┌────────────────────┐
                  │ PDF Document Store │
                  └────────────────────┘
```

---

## 📂 Repository Structure
```
knowledge_base_search_engine/
├── app.py                     # Streamlit frontend & RAG query pipeline
├── get_embedding_function.py  # Embedding generator using Ollama
├── populate_database.py       # CLI script to build/reset database
├── query_data.py              # CLI for querying data
├── requirements.txt           # Dependencies
└── README.md
```

---

## 🧩 Key Components

### **1. app.py**
- Main **Streamlit** interface.  
- Functions:  
  - Upload & process PDFs  
  - Query the knowledge base  
  - Display answers and document sources  
  - Manage database (clear/view stats)  
- Integrates **OllamaLLM (Mistral)** for response generation.

### **2. get_embedding_function.py**
- Uses `OllamaEmbeddings` with model `nomic-embed-text` to convert document chunks into vector representations.

### **3. populate_database.py**
- Command-line script to **load**, **split**, and **store** documents in **ChromaDB**.  
- Flags:  
  ```bash
  python populate_database.py --reset  # Clears DB and rebuilds
  ```

### **4. query_data.py**
- CLI interface to query stored knowledge base without UI.  
- Example:
  ```bash
  python query_data.py "What is the purpose of the project?"
  ```

---

## ⚙️ Installation & Setup

### **Step 1: Clone the Repository**
```bash
git clone https://github.com/<your-username>/knowledge_base_search_engine.git
cd knowledge_base_search_engine
```

### **Step 2: Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Step 3: Run Ollama**
Ensure Ollama is installed and models (`llama3`, `nomic-embed-text`) are available.
```bash
ollama pull llama3
ollama pull nomic-embed-text
```

### **Step 4: Launch the Web App**
```bash
streamlit run app.py
```

---

## 💡 Usage Guide

### **Upload Documents**
- Use the sidebar to upload PDFs.
- The system splits and embeds content into ChromaDB.

### **Ask Questions**
- Type a query in the “Ask a Question” box.
- The system retrieves top `k` relevant chunks and synthesizes a contextual answer.

### **View Chat History**
- Navigate to the “📊 Chat History” tab to view past queries and answers.

### **Manage Knowledge Base**
- View current documents and chunk count.
- Reset database if needed.

---

## 🧪 Example Query
**Question:**  
> “What is the main objective of this project?”

**Answer:**  
> The system is designed to search across documents and provide synthesized answers using LLM-based retrieval-augmented generation (RAG).

**Sources:**  
> - `Knowledge-base Search Engine.pdf (Page 2)`  
> - `Project_Overview.pdf (Page 1)`

---

## 🧰 Tech Stack
| Component | Tool |
|------------|------|
| Frontend | Streamlit |
| Backend | Python + LangChain |
| Vector Database | Pinecone |
| Embeddings | OllamaEmbeddings (`nomic-embed-text`) |
| LLM | OllamaLLM (`LLama-3`) |
| File Loader | LangChain’s PyPDFLoader |

---

## 🧾 Deliverables
- ✅ **GitHub Repository** (Open access)  
- ✅ **Demo Video (Optional)**  
- ✅ **Documentation (this file)**  

---

## 🧮 Evaluation Focus
| Criteria | Description |
|-----------|-------------|
| Retrieval Accuracy | Correctness of chunk retrieval |
| Synthesis Quality | Relevance and fluency of generated answers |
| Code Structure | Modularity, readability, maintainability |
| LLM Integration | Proper embedding and inference workflow |

---

## 🧑‍💻 Author
**N. Karthik Sagar**  
[GitHub](https://github.com/karthiksagarn) | [LinkedIn](https://linkedin.com/in/karthik-sagar-nallagula)
