# DocIntelli 📄🤖

DocIntelli is an intelligent PDF Chatbot that leverages **Retrieval-Augmented Generation (RAG)** to provide accurate, context-aware answers from your documents. Powered by **Google Gemini** and **LangChain**, it transforms static PDFs into interactive knowledge bases.

## 🚀 Key Features
- **Conversational AI:** Ask questions and get human-like responses.
- **Context-Aware:** Answers are grounded strictly in the content of your uploaded PDF.
- **Efficient Retrieval:** Uses **FAISS** (Facebook AI Similarity Search) for high-speed document indexing.
- **Scalable Pipeline:** Built with LangChain for modular and robust AI workflows.

## 🏗️ How It Works (RAG Architecture)
*(Diagram coming soon!)*

1. **Ingestion:** The system extracts text from your PDF using `PyPDF`.
2. **Chunking:** Text is split into small, overlapping segments to preserve context.
3. **Embeddings:** Chunks are converted into mathematical vectors using **Google Generative AI Embeddings**.
4. **Vector Store:** These vectors are stored in a **FAISS** index for semantic search.
5. **Retrieval & Generation:** When you ask a question, the system finds the most relevant chunks and sends them to **Google Gemini** to generate a final answer.

## 🛠️ Tech Stack
- **LLM:** Google Gemini (Gemini Pro)
- **Framework:** LangChain
- **Vector Database:** FAISS
- **Language:** Python 3.10+
- **Environment:** Dotenv for secure API management

## 📥 Installation & Setup

### 1. Clone the Project
```bash
git clone [https://github.com/Zakaria-png-tech/DocIntelli.git](https://github.com/Zakaria-png-tech/DocIntelli.git)
cd DocIntelli