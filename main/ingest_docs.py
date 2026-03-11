import os
from langchain_community.document_loaders import PyPDFLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

docs_path = "../docs"
persist_directory = "../vectorstore"

# 1. Load Documents
documents = []
if not os.path.exists(docs_path):
    print(f"Error: {docs_path} does not exist!")
else:
    for file in os.listdir(docs_path):
     if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(docs_path, file))
        loaded_docs = loader.load()
        print(f"Loaded {len(loaded_docs)} docs from {file}")
        documents.extend(loaded_docs)

if not documents:
    print("No PDFs found. Check your docs_path.")
else:
    # 2. Split Text
    # Chunk size 1000 with 200 overlap is a classic, reliable setting
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    # 3. Setup Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 4. Create and Save Vector Store
    # Note: .persist() is no longer needed in modern Chroma
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    print(f"Documents indexed! Created {len(chunks)} chunks in {persist_directory}")
