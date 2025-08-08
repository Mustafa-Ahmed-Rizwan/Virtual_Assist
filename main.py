import os
import json
import logging
import time
import re
import asyncio
from pathlib import Path
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from tiktoken import get_encoding
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
from rapidfuzz import process, fuzz

# Custom modules
from image_gen import generate_image
from suggestions_db import ENHANCED_SUGGESTIONS_DB_EN, ENHANCED_SUGGESTIONS_DB_DE
from trie_utils import OptimizedTrie

# ----------------- Configuration & Logging -----------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in .env")
if not openrouter_api_key:
    logger.warning("OPENROUTER_API_KEY not found in .env; ChatOpenAI will remain commented out")

# ----------------- Thread Pool for Blocking Operations -----------------
thread_pool = ThreadPoolExecutor(max_workers=4)

# ----------------- FastAPI Initialization -------------------
app = FastAPI(title="ARENA2036 Virtual Assistant")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Embeddings & Vector Stores ----------------
device = "cuda" if __import__('torch').cuda.is_available() else "cpu"

embeddings_en = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": device, "trust_remote_code": False}
)
embeddings_de = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    model_kwargs={"device": device, "trust_remote_code": False}
)

vectorstore_en = Chroma(
    persist_directory="vector_db",
    collection_name="arena2036_en",
    embedding_function=embeddings_en
)
vectorstore_de = Chroma(
    persist_directory="vector_db",
    collection_name="arena2036_de",
    embedding_function=embeddings_de
)
logger.info("Vector stores loaded successfully")

def get_vectorstore_and_embeddings(language: str):
    if language.upper() == 'DE':
        return vectorstore_de, embeddings_de
    return vectorstore_en, embeddings_en

# ----------------- Token Counting Utilities -----------------
enc = get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(enc.encode(text))

def log_request_size(contexts: List[str], question: str, template: str):
    q_toks = count_tokens(question)
    ctx_toks = sum(count_tokens(c) for c in contexts)
    overhead = count_tokens(template.format(context="{context}", question="{question}"))
    total = q_toks + ctx_toks + overhead
    logger.info(f"Token usage → question: {q_toks}, context: {ctx_toks}, overhead: {overhead}, total: {total}")
    return total
# ----------------- LLM Initialization -----------------------
temperature = float(os.getenv("LLM_TEMPERATURE", 0.0))
max_tokens = int(os.getenv("LLM_MAX_TOKENS", 1024))

# Groq-based LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=groq_api_key,
    temperature=temperature,
    max_tokens=max_tokens,
    top_p=0.95,
    timeout=60,
    max_retries=3
)

# OpenRouter-based LLM (commented out)
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

# ----------------- Prompt Template --------------------------
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are ARENA2036's intelligent and helpful virtual assistant. Using only the context provided below, generate a **clear**, **well-structured**, and **accurate** answer in proper **Markdown** format.

## 🧭 RESPONSE INSTRUCTIONS:
1. Use valid **Markdown formatting**
2. Use appropriate **headings (##)**
3. Use **bullet points**, **numbered lists**, or **tables** if helpful
4. Highlight **bold** and *italics*
5. Do **not** include any information not grounded in the provided context
6. If the question is not answerable with the given context but it is somewhat related, suggest visiting the website for more info
7. If unrelated, politely inform you cannot answer based on current information

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

# ----------------- Autocomplete Trie Setup ------------------
suggestion_trie_en = OptimizedTrie()
suggestion_trie_de = OptimizedTrie()
_valid_words_en = set()
_valid_words_de = set()

# Load English suggestions
for text, score in ENHANCED_SUGGESTIONS_DB_EN:
    suggestion_trie_en.insert(text, score)
    for w in re.findall(r"\b\w+\b", text):
        _valid_words_en.add(w)
_valid_words_en = list(_valid_words_en)

# Load German suggestions
for text, score in ENHANCED_SUGGESTIONS_DB_DE:
    suggestion_trie_de.insert(text, score)
    for w in re.findall(r"\b\w+\b", text):
        _valid_words_de.add(w)
_valid_words_de = list(_valid_words_de)

def get_autocomplete_suggestions(query: str, max_results: int = 20, language: str = "EN") -> List[str]:
    # Get the correct database based on language
    if language.upper() == "DE":
        db = ENHANCED_SUGGESTIONS_DB_DE
        trie = suggestion_trie_de
        valid_words = _valid_words_de
    else:
        db = ENHANCED_SUGGESTIONS_DB_EN
        trie = suggestion_trie_en
        valid_words = _valid_words_en
    
    if not query:
        return [item[0] for item in db[:max_results]]
    
    if len(query) < 2:
        # Return only language-specific suggestions
        return [item[0] for item in db if item[0].lower().startswith(query.lower())][:max_results]
    
    # Only use words from the correct language for correction
    tokens = re.findall(r"\b\w+\b|\W+", query)
    corrected = []
    for tok in tokens:
        if tok.isalnum():
            match = process.extractOne(tok, valid_words, scorer=fuzz.ratio)
            corrected.append(match[0] if match and match[1] >= 80 else tok)
        else:
            corrected.append(tok)
    corrected_query = "".join(corrected)
    
    # Search only in the language-specific trie
    suggestions = trie.search_prefix(corrected_query, max_results)
    
    # Ensure we only return suggestions that exist in our language DB
    db_texts = {item[0] for item in db}
    return [s for s in suggestions if s in db_texts][:max_results]

# ----------------- Image Intent Detection -------------------
image_templates = [
    # English - German
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
    "display a photo of",             

    # German versions
    "erstelle ein Bild von",
    "zeig mir ein Bild von",
    "erstelle eine Illustration von",
    "stelle ein Bild bereit",
    "zeichne mir ein Foto von",
    "rendere ein Bild von",
    "gib mir ein Bild von",
    "ich möchte ein Bild von",
    "erstelle eine Grafik von",
    "erzeuge eine visuelle Darstellung von",
    "skizziere mir eine Szene von",
    "zeige ein Foto von"
]

image_embs_en = embeddings_en.embed_documents(image_templates)
image_embs_de = embeddings_de.embed_documents(image_templates)
THRESHOLD = 0.6
_keyword_re = re.compile(r"\b(image|picture|photo|illustration|draw|show)\b", re.IGNORECASE)

def is_image_intent(text: str, language: str = "EN") -> bool:
    if _keyword_re.search(text):
        logger.info("Keyword match for image intent")
        return True
    emb_model, template_embs = (embeddings_de, image_embs_de) if language.upper()=="DE" else (embeddings_en, image_embs_en)
    user_emb = emb_model.embed_query(text)
    sims = cosine_similarity([user_emb], template_embs)[0]
    max_sim = float(np.max(sims))
    logger.info(f"Max image-intent similarity ({language}): {max_sim:.3f}")
    return max_sim >= THRESHOLD


# ----------------- Related Questions ------------------------
_sugg_cache = {"EN": None, "DE": None}

def get_suggestion_embeddings(lang: str = "EN") -> Tuple[np.ndarray, List[str]]:
    if lang.upper()=="DE":
        if _sugg_cache["DE"] is None:
            texts = [t for t,_ in ENHANCED_SUGGESTIONS_DB_DE]
            _sugg_cache["DE"] = (embeddings_de.embed_documents(texts), texts)
        return _sugg_cache["DE"]
    if _sugg_cache["EN"] is None:
        texts = [t for t,_ in ENHANCED_SUGGESTIONS_DB_EN]
        _sugg_cache["EN"] = (embeddings_en.embed_documents(texts), texts)
    return _sugg_cache["EN"]

def get_top_related_questions(user_question: str, limit: int = 5, language: str = "EN") -> List[str]:
    emb_model = embeddings_de if language.upper()=="DE" else embeddings_en
    user_emb = emb_model.embed_query(user_question)
    sugg_embs, texts = get_suggestion_embeddings(language)
    sims = cosine_similarity([user_emb], sugg_embs)[0]
    idxs = np.argsort(sims)[::-1][:limit]
    return [texts[i] for i in idxs if texts[i].strip().lower()!=user_question.strip().lower()][:limit]

# ----------------- Async Wrapper for Blocking Operations -----------------
async def run_qa_chain(qa_chain, query_text: str):
    """Run the QA chain in a thread pool to avoid blocking"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(thread_pool, qa_chain.invoke, {"query": query_text})

# ----------------- FastAPI Endpoints ------------------------
@app.get("/")
async def health_check():
    return {"status": "healthy"}

@app.get("/query")
async def query_assistant(request: Request, question: str, lang: str = "EN"):
    try:
        logger.info(f"Query: {question}, Language: {lang}")
        vectorstore, emb_model = get_vectorstore_and_embeddings(lang)

        if is_image_intent(question, lang):
            templates = image_templates
            embs = image_embs_de if lang.upper()=="DE" else image_embs_en
            user_emb = emb_model.embed_query(question)
            sims = cosine_similarity([user_emb], embs)[0]
            best_idx = int(np.argmax(sims))
            img_prompt = question.lower().replace(templates[best_idx], "").strip()
            if not img_prompt:
                raise HTTPException(status_code=400, detail="Image prompt empty")
            result = await generate_image(img_prompt)
            if result.get("error"):
                raise HTTPException(status_code=500, detail=result["error"])
            return {"answer": f"![Generated Image]({result['image_url']})", "sources": [], "is_image": True}

        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k":3, "fetch_k":12, "lambda_mult":0.9}
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template}
        )
        
        # typo correction
        def _correct(tok: str) -> str:
            # Use the appropriate word list based on language
            valid_words = _valid_words_de if lang.upper() == "DE" else _valid_words_en
            m = process.extractOne(tok, valid_words, scorer=fuzz.ratio)
            return m[0] if m and m[1]>=80 else tok
            
        tokens = re.findall(r"\b\w+\b|\W+", question)
        corrected = "".join([_correct(t) if t.isalnum() else t for t in tokens])
        if corrected != question:
            logger.info(f"Typo-corrected → '{corrected}'")
        query_text = corrected

        # FIXED: Use the async wrapper instead of awaiting the sync function
        res = await asyncio.wait_for(run_qa_chain(qa_chain, query_text), timeout=60.0)
        
        docs = res.get("source_documents", [])[:3]
        contexts = [d.page_content for d in docs]
        log_request_size(contexts, query_text, prompt_template.template)

        answer = res["result"]
        sources, seen = [], set()
        for d in docs:
            u = d.metadata.get("url", "")
            t = d.metadata.get("title", "Resource")
            if u and u not in seen:
                sources.append({"url": u, "title": t})
                seen.add(u)

        return {"answer": answer, "sources": sources, "is_image": False}

    except asyncio.TimeoutError:
        logger.error("Request timeout in /query")
        raise HTTPException(status_code=504, detail="Request timeout")
    except Exception as e:
        logger.error(f"Error in /query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/suggestions")
async def get_suggestions(q: str = "", limit: int = 20, lang: str = "EN"):
    try:
        start = time.time()
        sugg = get_autocomplete_suggestions(q, limit, lang)
        return {
            "suggestions": sugg,
            "query": q,
            "language": lang,
            "count": len(sugg),
            "processing_ms": round((time.time()-start)*1000,2)
        }
    except Exception as e:
        logger.error(f"Error getting suggestions: {e}")
        fallback = [t for t,_ in (ENHANCED_SUGGESTIONS_DB_DE if lang.upper()=="DE" else ENHANCED_SUGGESTIONS_DB_EN)][:limit]
        return {
            "suggestions": fallback,
            "query": q,
            "language": lang,
            "count": len(fallback),
            "error": "fallback"
        }

@app.get("/related-questions")
async def related_questions(question: str, limit: int = 5, lang: str = "EN"):
    try:
        related = get_top_related_questions(question, limit, lang)
        return {"related_questions": related}
    except Exception as e:
        logger.error(f"Error related-questions: {e}")
        return {"related_questions": []}

@app.post("/generate-image")
async def generate_image_endpoint(request: Request, prompt: str):
    try:
        if await request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")
        result = await generate_image(prompt)
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])
        return {"image_url": result["image_url"], "prompt": prompt}
    except Exception as e:
        logger.error(f"Error in generate-image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/serve-image")
async def serve_image(path: str):
    try:
        fp = Path(path).resolve()
        if not fp.exists():
            raise HTTPException(status_code=404, detail="Image file not found")
        return FileResponse(str(fp), media_type="image/png")
    except Exception as e:
        logger.error(f"Error serving image: {e}")
        raise HTTPException(status_code=500, detail="Failed to serve image")

# ----------------- Cleanup on shutdown -----------------
@app.on_event("shutdown")
async def shutdown_event():
    thread_pool.shutdown(wait=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)