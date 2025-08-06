import os
from fastapi import FastAPI, HTTPException, Request
import asyncio
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from langchain_groq import ChatGroq  # Commented out Groq import
from langchain_chroma import Chroma
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
from image_gen import generate_image
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from tiktoken import get_encoding
# New import for OpenRouter
from langchain_openai import ChatOpenAI
from suggestions_db import ENHANCED_SUGGESTIONS_DB  # Import the suggestions database
import re
from rapidfuzz import process, fuzz

# Build a set of every word in your suggestion-DB (for typo lookup)
_valid_words = set()
for text, _ in ENHANCED_SUGGESTIONS_DB:
     for w in re.findall(r"\b\w+\b", text):
         _valid_words.add(w)
_valid_words = list(_valid_words)


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")  # Commented out Groq API key
# openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
# if not openrouter_api_key:
#     raise ValueError("OPENROUTER_API_KEY not found in .env file")

# Initialize FastAPI app
app = FastAPI(title="ARENA2036 Virtual Assistant")

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
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = Chroma(
        persist_directory="arena_vectordb", #vector_db_arena #vector_data
        collection_name="arena2036_en",
        embedding_function=embeddings
    )
    logger.info("Vector store loaded successfully")
except Exception as e:
    logger.error(f"Failed to load vector store: {str(e)}")
    raise HTTPException(status_code=500, detail="Vector store initialization failed")

# --- TOKEN COUNTING UTILITIES ---
enc = get_encoding("cl100k_base")  # adjust encoding as needed

def count_tokens(text: str) -> int:
    return len(enc.encode(text))

def log_request_size(contexts: List[str], question: str, template: str):
    q_toks = count_tokens(question)
    ctx_toks = sum(count_tokens(c) for c in contexts)
    overhead = count_tokens(template.format(context="{context}", question="{question}"))
    total = q_toks + ctx_toks + overhead
    logger.info(f"Token usage → question: {q_toks}, context: {ctx_toks}, overhead: {overhead}, total: {total}")
    return total

# Initialize LLM with optimized settings
# — Production LLM settings —
temperature = float(os.getenv("LLM_TEMPERATURE", 0.0))  # low creativity
max_tokens = int(os.getenv("LLM_MAX_TOKENS", 1024))       # cap response size
# Commented out original Groq LLM initialization
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=groq_api_key,
    temperature=temperature,
    max_tokens=max_tokens,
    top_p=0.95,
    timeout=60,
    max_retries=3
)
# New OpenRouter LLM initialization
# llm = ChatOpenAI(
#     model="mistralai/mistral-small-3.2-24b-instruct:free",
#     api_key=openrouter_api_key,
#     base_url="https://openrouter.ai/api/v1",
#     temperature=temperature,
#     max_tokens=max_tokens,
#     top_p=0.9,
#     timeout=60,
#     max_retries=2
# )
logger.info("LLM initialized successfully")

# Main QA prompt template
from langchain.prompts import PromptTemplate

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are ARENA2036's intelligent and helpful virtual assistant. Using only the context provided below, generate a **clear**, **well-structured**, and **accurate** answer in proper **Markdown** format.

## 🧭 RESPONSE INSTRUCTIONS:

- Use valid **Markdown formatting**
- Use appropriate **headings (##)** to organize sections logically
- Use **bullet points**, **numbered lists**, or **tables** if it improves readability
- Highlight key terms using **bold** and important concepts using *italics*
- Do **not** include any information not grounded in the provided context

---

### 📚 Context:
{context}

---

### ❓ User Question:
{question}

---

### ✅ Answer:
"""
)

# — Production retriever settings —
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,        
        "fetch_k": 12,    
        "lambda_mult": 0.9
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

image_templates = [
    "generate an image of",
    "show me a picture of",
    "create an illustration of",
    "provide an image",
    "draw me a photo of",
    "render an image of",
    "give me a picture of",
    "i want an image of",
    "make a graphic of",
    "produce a visual of",
    "sketch me a scene of",
    "display a photo of"
]

image_embs = embeddings.embed_documents(image_templates)
THRESHOLD = 0.6  # lower threshold for recall
# simple keyword fallback
_keyword_re = re.compile(r"\b(image|picture|photo|illustration|draw|show)\b", re.IGNORECASE)
def is_image_intent(text: str) -> bool:
    # quick keyword check
    if _keyword_re.search(text):
        logger.info("Keyword match for image intent")
        return True
    # embedding check
    user_emb = embeddings.embed_query(text)
    sims = cosine_similarity([user_emb], image_embs)[0]
    max_sim = float(np.max(sims))
    logger.info(f"Max image-intent similarity: {max_sim}")
    return max_sim >= THRESHOLD

# API Endpoints
@app.get("/")
async def health_check():
    return {"status": "healthy"}

# Modify the existing /query endpoint to handle image generation
@app.get("/query")
async def query_assistant(request: Request, question: str):
    try:
        logger.info(f"Query: {question}")
        if await request.is_disconnected():
            return {"error": "Request aborted"}
         # —— word-level typo correction (text QA only) ——
        def _correct_word(tok: str) -> str:
            match = process.extractOne(tok, _valid_words, scorer=fuzz.ratio)
            return match[0] if match and match[1] >= 80 else tok

    # split into words & non-words to keep punctuation
        tokens = re.findall(r"\b\w+\b|\W+", question)
        corrected_tokens = [
         _correct_word(t) if t.isalnum() else t
         for t in tokens
       ]
        corrected_question = "".join(corrected_tokens)
        if corrected_question != question:
          logger.info(f"Typo-corrected → '{corrected_question}'")
        else:
          corrected_question = question

        # Use embedding-based intent detection
        if is_image_intent(question):
            # find best template match to strip prefix
            user_emb = embeddings.embed_query(question)
            sims = cosine_similarity([user_emb], image_embs)[0]
            best_idx = int(np.argmax(sims))
            image_prompt = question.lower().replace(image_templates[best_idx], "").strip()
            if not image_prompt:
                raise HTTPException(status_code=400, detail="Image generation prompt is empty")

            result = await generate_image(image_prompt)
            if result.get("error"):
                raise HTTPException(status_code=500, detail=result["error"])

            return {
                "answer": f"![Generated Image]({result['image_url']})",
                "sources": [],
                "is_image": True
            }

        # Regular QA flow
        async def run_qa_chain():
            return qa_chain.invoke({"query": corrected_question})

        res = await asyncio.wait_for(run_qa_chain(), timeout=60.0)
        # LOG token usage
        docs = res.get("source_documents", [])[:3]
        contexts = [doc.page_content for doc in docs]
        log_request_size(contexts, corrected_question, prompt_template.template)
        if await request.is_disconnected():
            return {"error": "Request aborted"}

        answer = res["result"]
        docs = res.get("source_documents", [])[:3]
        sources, seen = [], set()
        for doc in docs:
            url = doc.metadata.get("url", "")
            title = doc.metadata.get("title", "Resource")
            if url and url not in seen:
                sources.append({"url": url, "title": title})
                seen.add(url)

        return {"answer": answer, "sources": sources, "is_image": False}

    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timeout")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/serve-image")
async def serve_image(path: str):
    import os
    from fastapi.responses import FileResponse
    try:
        file_path = os.path.abspath(path)  # Validate path
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Image not found")
        return FileResponse(file_path, media_type="image/png")  # Adjust media type if needed
    except Exception as e:
        logger.error(f"Error serving image: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to serve image")

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

@app.post("/generate-image")
async def generate_image_endpoint(request: Request, prompt: str):
    """Generate an image based on the provided prompt."""
    try:
        logger.info(f"Image generation request: {prompt}")
        
        # Check if client disconnected
        if await request.is_disconnected():
            logger.info("Client disconnected, aborting image generation")
            return {"error": "Request aborted"}
        
        # Generate image
        result = await generate_image(prompt)
        
        if result["error"]:
            raise HTTPException(status_code=500, detail=result["error"])
            
        return {
            "image_url": result["image_url"],
            "prompt": prompt
        }
        
    except asyncio.CancelledError:
        logger.info("Image generation cancelled by client")
        return {"error": "Request cancelled"}
    except Exception as e:
        logger.error(f"Image generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/related-questions")
async def get_related_questions(question: str, limit: int = 5):
    """Return related questions most similar to the user's question using embeddings,
       but never echo back the exact question."""
    # get a few extra so we can filter one out
    raw = get_top_related_questions(question, limit + 1)
    # filter out any exact (case-insensitive) match
    filtered = [
        q for q in raw
        if q.strip().lower() != question.strip().lower()
    ]
    # return up to `limit`
    return {"related_questions": filtered[:limit]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 