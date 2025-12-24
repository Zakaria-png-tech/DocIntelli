# DocIntelli 📄🤖

DocIntelli is an intelligent PDF Chatbot that leverages **Retrieval-Augmented Generation (RAG)** to provide accurate, context-aware answers from your documents. Powered by **Google Gemini** and **LangChain**, it features a sleek **Streamlit** web interface for seamless document interaction.

## 🚀 Key Features
- **Interactive Web UI:** Simple drag-and-drop PDF upload via Streamlit.
- **Conversational AI:** Ask questions and get human-like responses in a chat interface.
- **Context-Aware:** Answers are grounded strictly in the content of your uploaded PDF.
- **Efficient Retrieval:** Uses **FAISS** for high-speed document indexing.

## 🏗️ How It Works (RAG Architecture)
*(Diagram coming soon!)*

1. **Ingestion:** Upload a PDF; the system extracts text using `PyPDF`.
2. **Chunking:** Text is split into small segments to preserve context.
3. **Embeddings:** Chunks are converted into vectors using **Google Generative AI Embeddings**.
4. **Vector Store:** Stored in a **FAISS** index for semantic search.
5. **Retrieval & Generation:** Gemini generates a final answer using the most relevant chunks.

## 🛠️ Tech Stack
- **Frontend/UI:** Streamlit
- **LLM:** Google Gemini (Gemini Pro)
- **Framework:** LangChain
- **Vector Database:** FAISS
- **Language:** Python 3.10+

## 📥 Installation & Setup

### 1. Clone the Project
```bash
git clone [https://github.com/Zakaria-png-tech/DocIntelli.git](https://github.com/Zakaria-png-tech/DocIntelli.git)
cd DocIntelli