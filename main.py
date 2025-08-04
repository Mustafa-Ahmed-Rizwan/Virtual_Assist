import os
from fastapi import FastAPI, HTTPException, Request
import asyncio
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import logging
import json
import random
import time
from collections import deque
from typing import List
from trie_utils import TrieNode, OptimizedTrie

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

# Initialize FastAPI app
app = FastAPI(title="Arena2036 Virtual Assistant")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize embeddings and vector store
try:
    embeddings = HuggingFaceEmbeddings( model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = Chroma(
        persist_directory="vector_db",
        collection_name="arena2036_en",
        embedding_function=embeddings
    )
    logger.info("Vector store loaded successfully")
except Exception as e:
    logger.error(f"Failed to load vector store: {str(e)}")
    raise HTTPException(status_code=500, detail="Vector store initialization failed")

# Initialize LLM with optimized settings
temperature = float(os.getenv("LLM_TEMPERATURE", 0.0))  # Deterministic by default
max_tokens = int(os.getenv("LLM_MAX_TOKENS", 500))
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=groq_api_key,
    temperature=temperature,
    max_tokens=max_tokens,
    top_p=0.9,
    timeout=60,
    max_retries=5
)
logger.info("LLM initialized successfully")

# Main QA prompt template
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are ARENA2036's helpful virtual assistant. Provide clear, focused answers using the context provided.

RESPONSE GUIDELINES:
1. Answer directly and specifically - focus on what the user asked
2. Use simple, clear language and proper Markdown formatting
3. Structure information logically with headers (##) and bullet points when needed
4. Provide essential information without overwhelming details
5. Include only the most relevant steps or actions
6. Keep responses concise but complete
7. Use **bold** for important terms and *italics* for emphasis

Context Information:
{context}

User Question: {question}

Helpful Answer:"""
)



# Configure retriever parameters tuned to your vector DB
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,         # return top 4 chunks
        "fetch_k": 5,   # consider top 8 for MMR
        "lambda_mult": 0.6
    }
)

# Initialize QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt_template}
)
logger.info("QA chain created with optimized retrieval settings")

# Enhanced suggestions database with scoring
ENHANCED_SUGGESTIONS_DB = [
    ("How do I connect my domain to ARENA2036?", 0.9),
    ("How do I set up ARENA2036 Services?", 0.95),
    ("How do I use ARENA2036 Projects?", 0.9),
    ("How do I reset my ARENA2036 account password?", 0.8),
    ("How do I customize my ARENA2036 profile?", 0.7),
    ("How to configure ARENA2036 settings?", 0.8),
    ("How to integrate ARENA2036 with third-party tools?", 0.85),
    ("How to manage ARENA2036 notifications?", 0.7),
    ("How to export data from ARENA2036?", 0.75),
    ("How to collaborate in ARENA2036 Projects?", 0.8),
    ("How to backup ARENA2036 data?", 0.7),
    ("How to upgrade ARENA2036 subscription?", 0.6),
    ("How to delete ARENA2036 account?", 0.5),
    ("How to contact ARENA2036 support?", 0.8),
    ("What are ARENA2036 system requirements?", 0.6),
    ("How to troubleshoot ARENA2036 login issues?", 0.85),
    ("How to share ARENA2036 projects?", 0.8),
    ("How to use ARENA2036 API?", 0.7),
    ("How to install ARENA2036 desktop app?", 0.75),
    ("How to recover deleted ARENA2036 files?", 0.8),
    ("ARENA2036 pricing plans comparison", 0.6),
    ("ARENA2036 security features overview", 0.7),
    ("How to migrate from other platforms to ARENA2036?", 0.65),
    ("ARENA2036 mobile app download", 0.7),
    ("How to create ARENA2036 workspace?", 0.8),
    ("ARENA2036 keyboard shortcuts list", 0.6),
    ("How to enable two-factor authentication ARENA2036?", 0.75),
    ("ARENA2036 data synchronization issues", 0.7),
    ("How to invite team members to ARENA2036?", 0.8),
    ("ARENA2036 file sharing permissions", 0.7),
    # Website research topics
    ("What is Wire Harness Automation and Standardization?", 0.8),
    ("What is Industrial AI?", 0.8),
    ("What is the Asset Administration Shell (AAS)?", 0.8),
    ("What is the Industrial Metaverse?", 0.8),
    ("What are Data Spaces?", 0.8),
    ("Where can I find ARENA2036 publications?", 0.7),
    # Project names
    ("What is the Well-defined Research Campus Initiative?", 0.7),
    ("What is CARpulse?", 0.7),
    ("What is EcoFrame?", 0.7),
    ("What is Connect4HCA?", 0.7),
    ("What is ARENA2036-X?", 0.7),
    ("What is the Network infrastructure for Industry 4.0?", 0.7),
    ("What is Catena-X?", 0.7),
    ("What is DigiTain?", 0.7),
    ("What is the Interactive Bosch floor?", 0.7),
    ("What is the Wire Harness standardization initiative?", 0.7),
    ("What is the Transformation hub for the Wire Harness?", 0.7),
    ("What is the Asset Administration Shell for the Wire Harness?", 0.7),
    ("How to view all ARENA2036 projects?", 0.6),
    ("How to view completed ARENA2036 projects?", 0.6)
]

# Initialize optimized trie
suggestion_trie = OptimizedTrie()
for suggestion, score in ENHANCED_SUGGESTIONS_DB:
    suggestion_trie.insert(suggestion, score)

def get_autocomplete_suggestions(query: str, max_results: int = 20) -> List[str]:
    """Ultra-fast autocomplete using optimized Trie."""
    if not query:
        return [item[0] for item in ENHANCED_SUGGESTIONS_DB[:max_results]]
    
    if len(query) < 2:
        # Return suggestions that start with the character
        filtered = [item[0] for item in ENHANCED_SUGGESTIONS_DB 
                   if item[0].lower().startswith(query.lower())]
        return filtered[:max_results]
    
    return suggestion_trie.search_prefix(query, max_results)

# API Endpoints
@app.get("/")
async def health_check():
    return {"status": "healthy"}

@app.get("/query")
async def query_assistant(request: Request, question: str):
    try:
        logger.info(f"Query: {question}")
        
        # Check if client disconnected
        if await request.is_disconnected():
            logger.info("Client disconnected, aborting query")
            return {"error": "Request aborted"}
        
        # Create a timeout for the LLM call
        async def run_qa_chain():
            return qa_chain({"query": question})
        
        try:
            # Run with timeout and check for disconnection
            res = await asyncio.wait_for(run_qa_chain(), timeout=60.0)
            
            # Check again if client disconnected during processing
            if await request.is_disconnected():
                logger.info("Client disconnected during processing")
                return {"error": "Request aborted"}
                
        except asyncio.TimeoutError:
            logger.warning(f"Query timeout for: {question}")
            raise HTTPException(status_code=504, detail="Request timeout")
        
        answer = res["result"]
        docs = res.get("source_documents", [])[:3]

        sources = []
        seen_urls = set()
        for doc in docs:
            url = doc.metadata.get("url", "")
            title = doc.metadata.get("title", "Resource")
            if url and url not in seen_urls:
                sources.append({"url": url, "title": title})
                seen_urls.add(url)

        return {"answer": answer, "sources": sources}
        
    except asyncio.CancelledError:
        logger.info("Query cancelled by client")
        return {"error": "Request cancelled"}
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/suggestions")
async def get_suggestions(q: str = "", limit: int = 20):
    """Get autocomplete suggestions with optional query and limit."""
    try:
        start_time = time.time()
        suggestions = get_autocomplete_suggestions(q, limit)
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return {
            "suggestions": suggestions,
            "query": q,
            "count": len(suggestions),
            "processing_time_ms": round(processing_time, 2)
        }
    except Exception as e:
        logger.error(f"Error getting suggestions: {str(e)}")
        fallback_suggestions = [item[0] for item in ENHANCED_SUGGESTIONS_DB[:limit]]
        return {
            "suggestions": fallback_suggestions,
            "query": q,
            "count": len(fallback_suggestions),
            "processing_time_ms": 0,
            "error": "Using fallback suggestions"
        }


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Precompute embeddings for suggestions (cache at startup)
_suggestion_texts = [item[0] for item in ENHANCED_SUGGESTIONS_DB]
_suggestion_embeddings = None
def get_suggestion_embeddings():
    global _suggestion_embeddings
    if _suggestion_embeddings is None:
        _suggestion_embeddings = embeddings.embed_documents(_suggestion_texts)
    return _suggestion_embeddings

def get_top_related_questions(user_question: str, limit: int = 4):
    try:
        user_emb = embeddings.embed_query(user_question)
        sugg_embs = get_suggestion_embeddings()
        # Compute cosine similarity
        sims = cosine_similarity([user_emb], sugg_embs)[0]
        # Get indices of top N
        top_idx = np.argsort(sims)[::-1][:limit]
        return [_suggestion_texts[i] for i in top_idx]
    except Exception as e:
        logger.error(f"Related question similarity error: {str(e)}")
        # Fallback: return top suggestions
        return _suggestion_texts[:limit]

@app.get("/related-questions")
async def get_related_questions(question: str, limit: int = 4):
    """Return related questions most similar to the user's question using embeddings."""
    related = get_top_related_questions(question, limit)
    return {"related_questions": related}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
