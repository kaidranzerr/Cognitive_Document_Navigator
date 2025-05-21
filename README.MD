# 📄 Chat With Documents — AI-Powered PDF Document QA System

This application allows users to **upload PDF documents** and **chat with them** using **OpenAI GPT-4o-mini**. The system extracts factual chunks from documents, embeds them using OpenAI embeddings, and provides **context-aware responses** grounded in document content.

---

## 🔧 Features

### 🔍 Chat Interface (`chat.py`)
- Uses Streamlit for a simple, elegant frontend.
- Embeds the user's query and retrieves **top-5 most similar chunks** from uploaded documents using PostgreSQL vector similarity.
- Sends a system prompt along with relevant document chunks to GPT-4o-mini.
- Displays references for transparency.
- Chat history is stored using `st.session_state`.

### 📤 Document Upload & Management (`manage_documents.py`)
- Upload PDF files and automatically:
  - Extract text using `pdftotext`
  - Split into chunks (~4000 characters)
  - Generate factual summaries using OpenAI
  - Store facts with vector embeddings
  - Detect relevant tags via OpenAI prompt
- Tag system to categorize documents.
- Displays uploaded documents and allows deletion.

### 🏷️ Tag Management (`tags.py`)
- Add, view, and delete document tags through a simple UI.

---

## 🧠 How It Works

1. **PDF Upload:**
   - Text is extracted from PDF.
   - Text is chunked (~4000 characters).
   - Each chunk is processed by GPT-4o-mini to extract factual statements.
   - Factual chunks are embedded and saved in PostgreSQL.

2. **Query Processing:**
   - User query is embedded.
   - Top-5 similar chunks are retrieved based on cosine similarity (`embedding <-> ai.openai_embed(...)`).
   - Chunks are included in a prompt and sent to GPT-4o-mini.
   - Response is generated and shown with document references.

3. **Tags:**
   - Existing tags are matched to document content using a prompt to GPT.
   - Useful for organizing and searching uploaded files.

---

## 🗃️ Tech Stack

- **Frontend**: Streamlit
- **LLM & Embeddings**: OpenAI GPT-4o-mini, text-embedding-3-small
- **Database**: PostgreSQL (via Peewee ORM)
- **PDF Parsing**: `pdftotext`
- **Async Runtime**: asyncio + anyio

---

## 📂 File Structure

