# Ethiopian Constitution RAG

A Retrieval-Augmented Generation (RAG) application that lets you ask questions about the Ethiopian Constitution and get accurate, context-aware answers with page references.

## Architecture

```
PDF → Chunking → HuggingFace Embeddings → Qdrant Vector DB
                                                    ↓
User Query → Similarity Search → Context + Gemini LLM → Answer
```

**Stack:**
- **Embeddings:** `BAAI/bge-small-en-v1.5` (local, free)
- **Vector DB:** Qdrant (Docker)
- **LLM:** Google Gemini via OpenAI-compatible API
- **Framework:** LangChain

## Prerequisites

- Python 3.10+
- Docker
- A Google AI API key ([get one here](https://aistudio.google.com/apikey))

## Setup

**1. Clone the repo**

```bash
git clone https://github.com/bekalu73/ethiopian-constitution-rag.git
cd ethiopian-constitution-rag
```

**2. Start Qdrant**

```bash
docker compose up -d
```

**3. Install dependencies**

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**4. Configure environment**

```bash
cp .env.example .env
```

Add your Gemini API key to `.env`:

```
GEMINI_API_KEY=your_api_key_here
```

## Usage

**Step 1 — Index the constitution**

```bash
python index.py
```

This loads the PDF, splits it into chunks, generates embeddings, and stores them in Qdrant.

**Step 2 — Start chatting**

```bash
python chat.py
```

Type your question and get an answer with page references.

## Project Structure

```
├── chat.py                     # Interactive Q&A script
├── index.py                    # PDF indexing script
├── Ethiopia_Constitution.pdf   # Source document
├── docker-compose.yml          # Qdrant service
├── requirements.txt            # Python dependencies
└── .gitignore
```

## How It Works

1. **Indexing** (`index.py`) — Loads the Ethiopian Constitution PDF, splits it into 1000-character chunks with 200-character overlap, generates embeddings using HuggingFace, and stores them in Qdrant.

2. **Querying** (`chat.py`) — Takes a user question, performs similarity search against the vector store, retrieves relevant chunks with page numbers, and sends them as context to Gemini to generate an answer.

## License

MIT
