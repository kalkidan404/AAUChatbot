# General AAU RAG Assistant

## Objective

Build a general RAG assistant that answers questions about Addis Ababa University using documents.

## Running the Project

From the project root (`AAUchatbot`):

Install dependencies:

```bash
pip install -r shared_starter_code/requirements.txt
```

Run backend:

```bash
uvicorn shared_starter_code.app:app --reload
```

Run frontend:

```bash
streamlit run shared_starter_code/chat.py
```

## Uploading Documents

Place PDF/TXT documents inside the `docs` folder.

## Asking Questions

Ask questions in natural language.  
The assistant will respond based only on the documents.

## Optional Features

- Highlight retrieved text
- Summarization
- Multi-file retrieval
- Conversation memory
