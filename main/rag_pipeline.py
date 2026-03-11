import os
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate

# For cloud loading
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# 1. Setup Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Load Vector Store
# 2. Load Vector Store
def initialize_db():
    if os.path.exists("./docs"):
        # Use a more aggressive glob to find EVERYTHING
        loader = PyPDFDirectoryLoader("./docs", glob="**/[!.]*.pdf") 
        docs = loader.load()
        
        # Log which files were actually loaded
        loaded_sources = list(set([d.metadata.get("source") for d in docs]))
        print(f"[DEBUG] Successfully loaded pages from: {loaded_sources}")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)
        return Chroma.from_documents(documents=splits, embedding=embeddings)
    return None

vectordb = initialize_db()

# 3. Setup Retriever - INCREASE K TO 20
retriever = vectordb.as_retriever(search_kwargs={"k": 20}) 


# 4. Initialize LLM
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0
)

# 5. Refined Prompt
template = """You are a helpful assistant. Use the following pieces of retrieved context to answer the question. 
If the answer is not contained within the context, honestly state that you do not know. 

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
    result = qa_chain.invoke({"query": question})
    answer = result["result"]
    source_docs = result["source_documents"]
    
    # Extract filenames only (e.g., Handbook.pdf)
    sources = list(set([os.path.basename(doc.metadata.get("source", "unknown")) for doc in source_docs]))
    chunks = [doc.page_content for doc in source_docs]

    # Added your debug logic here:
    if "don't know" in answer.lower() or "do not know" in answer.lower():
        print(f"\n[DEBUG] Found {len(source_docs)} documents, but LLM couldn't find the answer in them.")

    return answer, sources, chunks
