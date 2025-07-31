import os
import json
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# Configuration based on stats
CHUNK_SIZE = 10000       # characters per chunk
CHUNK_OVERLAP = 500      # characters overlap
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
PERSIST_DIRECTORY = 'vector_db'
COLLECTION_NAME = 'arena2036_en'

INPUT_FILE = 'arena_data_en.jsonl'

# Ensure output directory
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

# Load all documents without filtering short content
documents = []
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        rec = json.loads(line)
        url = rec.get('url', '')
        title = rec.get('title', '')
        content = rec.get('content', '') or ''
        # Include all content, even very short entries
        metadata = {'url': url, 'title': title}
        documents.append(Document(page_content=content, metadata=metadata))

print(f"Loaded {len(documents)} documents.")

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len
)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")

# Initialize embeddings and vectorstore
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory=PERSIST_DIRECTORY,
    collection_name=COLLECTION_NAME
)

# Add to vectorstore
vectorstore.add_documents(chunks)
vectorstore.persist()

print(f"Vector DB created at '{PERSIST_DIRECTORY}', collection '{COLLECTION_NAME}'.")