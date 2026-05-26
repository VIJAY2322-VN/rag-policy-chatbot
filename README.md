# 🛡️ PolicyExpert AI

**PolicyExpert AI** is a premium Retrieval-Augmented Generation (RAG) chatbot designed to help organizations navigate complex policy documents, compliance guidelines, and internal handbooks with ease.

Built with a high-end interface and a robust AI engine, it provides instant, source-backed answers to your most critical policy questions.

## ✨ Key Features

*   **💎 Premium UI/UX**: Modern dark-themed interface with glassmorphism, fluid animations, and high-end typography.
*   **🔍 Intelligent Retrieval**: Optimized RAG engine using FAISS and HuggingFace embeddings for precise context matching.
*   **📑 Multi-Document Support**: Seamlessly process and index PDF and Text documents.
*   **🔗 Source Attribution**: Every answer includes expandable "Referenced Sources" showing exact document snippets.
*   **⚡ dual-LLM Support**: Compatibility with both **OpenAI (GPT-3.5/4)** and **Groq (Mixtral/Llama 3)** for lightning-fast responses.
*   **🔄 Instant Sync**: Auto-index existing documents in the `data/` directory with a single click.

## 🛠️ Tech Stack

- **Frontend**: Streamlit (Premium Custom CSS)
- **RAG Framework**: LangChain
- **Vector Database**: FAISS
- **Embeddings**: HuggingFace (`all-MiniLM-L6-v2`)
- **LLM Providers**: OpenAI & Groq

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- OpenAI or Groq API Key

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/VIJAY2322-VN/rag-policy-chatbot.git
   cd rag-policy-chatbot
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python -m streamlit run app.py
   ```

## 📂 Project Structure

```text
rag-policy-chatbot/
├── app.py              # Main Streamlit application
├── data/               # Document storage (policy.txt, etc.)
├── src/
│   ├── engine.py       # RAG logic & Document Management
│   └── styles.py       # Premium CSS & UI enhancements
└── requirements.txt    # Project dependencies
```

## 🛡️ License
This project is licensed under the MIT License - feel free to use it for your organization!

