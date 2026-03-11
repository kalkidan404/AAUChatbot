import os
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate

load_dotenv()

# 1. Setup Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Load Vector Store
vectordb = Chroma(
    persist_directory="../vectorstore",
    embedding_function=embeddings
)

# 3. Setup Retriever
retriever = vectordb.as_retriever(search_kwargs={"k": 50})

# 4. Initialize LLM (Groq Llama 3.1)
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0
)

# 5. Refined Prompt
template = """You are a helpful assistant. Use the following pieces of retrieved context to answer the question. 
If the answer is not contained within the context, honestly state that you do not know. 
Do not make up information outside of the provided context.

Context:
{context}

Question: {question}

Answer:"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# 6. Create RAG Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

def ask(question):
    # Use .invoke() for modern compatibility
    result = qa_chain.invoke({"query": question})

    answer = result["result"]
    source_docs = result["source_documents"]

    # Extract unique source names/metadata
    sources = list(set([doc.metadata.get("source", "unknown") for doc in source_docs]))

    # Also return the retrieved text for frontend display
    chunks = [doc.page_content for doc in source_docs]

    # Debug print: If answer is "I don't know", check if source_docs is empty
    if "don't know" in answer.lower() or "do not know" in answer.lower():
        print(f"\n[DEBUG] Found {len(source_docs)} documents, but LLM couldn't find the answer in them.")

    return answer, sources, chunks

# Example Usage:
# response, docs, texts = ask("What is the main topic of the document?")
# print(f"Answer: {response}\nSources: {docs}\nRetrieved Texts: {texts}")