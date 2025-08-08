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
from langchain.chains.question_answering import load_qa_chain
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
    persist_directory="arena_vector_database",
    collection_name="arena2036_en",
    embedding_function=embeddings_en
)
vectorstore_de = Chroma(
    persist_directory="arena_vector_database",
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

        # Initialize token counters
        token_usage = {
            "input": {
                "prompt_template": 0,
                "question": 0,
                "contexts": 0,
                "total": 0
            },
            "output": 0,
            "total": 0
        }

        # Image generation path
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
            
            # Count tokens for image path (minimal)
            token_usage["input"]["question"] = count_tokens(question)
            token_usage["input"]["total"] = token_usage["input"]["question"]
            token_usage["total"] = token_usage["input"]["total"]
            
            logger.info(
                f"Token Usage (Image Request):\n"
                f"  INPUT: {token_usage['input']['total']} tokens\n"
                f"    - Question: {token_usage['input']['question']}\n"
                f"  OUTPUT: Image generated (token count not applicable)"
            )
            
            return {
                "answer": f"![Generated Image]({result['image_url']})", 
                "sources": [], 
                "is_image": True,
                "token_usage": token_usage
            }

        # Setup retriever
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k":3, "fetch_k":12, "lambda_mult":0.9}
        )
        
        # Typo correction
        def _correct(tok: str) -> str:
            valid_words = _valid_words_de if lang.upper() == "DE" else _valid_words_en
            m = process.extractOne(tok, valid_words, scorer=fuzz.ratio)
            return m[0] if m and m[1]>=80 else tok
            
        tokens = re.findall(r"\b\w+\b|\W+", question)
        corrected = "".join([_correct(t) if t.isalnum() else t for t in tokens])
        if corrected != question:
            logger.info(f"Typo-corrected → '{corrected}'")
        query_text = corrected

        # Get documents and apply smart context truncation
        docs = retriever.get_relevant_documents(query_text)
        contexts = [d.page_content for d in docs]
        
        # Apply smart context truncation
        contexts = smart_context_truncation(contexts, query_text, emb_model, max_context_tokens=4000)
        
        # Update documents with truncated contexts
        docs = docs[:len(contexts)]  # Ensure doc count matches context count
        for i, ctx in enumerate(contexts):
            if i < len(docs):
                docs[i].page_content = ctx

        # Create QA chain with truncated contexts
        from langchain.chains.question_answering import load_qa_chain
        qa_chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt_template)

        # Execute QA chain
        start_time = time.time()
        res = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                thread_pool, 
                lambda: qa_chain({"input_documents": docs, "question": query_text})
            ), 
            timeout=60.0
        )
        processing_time = time.time() - start_time
        
        # Process results
        answer = res["output_text"]
        
        # Calculate token usage
        token_usage["input"]["prompt_template"] = count_tokens(prompt_template.template)
        token_usage["input"]["question"] = count_tokens(query_text)
        token_usage["input"]["contexts"] = sum(count_tokens(c) for c in contexts)
        token_usage["input"]["total"] = (
            token_usage["input"]["prompt_template"] +
            token_usage["input"]["question"] +
            token_usage["input"]["contexts"]
        )
        
        token_usage["output"] = count_tokens(answer)
        token_usage["total"] = token_usage["input"]["total"] + token_usage["output"]
        
        # Detailed logging
        logger.info(
            f"Token Usage (LLM Request):\n"
            f"  INPUT: {token_usage['input']['total']} tokens\n"
            f"    - Template: {token_usage['input']['prompt_template']}\n"
            f"    - Question: {token_usage['input']['question']}\n"
            f"    - Contexts: {token_usage['input']['contexts']}\n"
            f"  OUTPUT: {token_usage['output']} tokens\n"
            f"  TOTAL: {token_usage['total']} tokens\n"
            f"Processing time: {processing_time:.2f}s"
        )

        # Prepare sources
        sources, seen = [], set()
        for d in docs:
            u = d.metadata.get("url", "")
            t = d.metadata.get("title", "Resource")
            if u and u not in seen:
                sources.append({"url": u, "title": t})
                seen.add(u)

        return {
            "answer": answer,
            "sources": sources,
            "is_image": False,
            "processing_time": processing_time,
            "token_usage": token_usage
        }

    except asyncio.TimeoutError:
        logger.error("Request timeout in /query")
        raise HTTPException(status_code=504, detail="Request timeout")
    except Exception as e:
        logger.error(f"Error in /query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


def smart_context_truncation(contexts: List[str], question: str, embeddings_model, max_context_tokens: int = 4000) -> List[str]:
    """
    Intelligently truncate context based on relevance to the question.
    Only truncates when necessary and prioritizes the most relevant content.
    
    Args:
        contexts: List of context strings from retriever
        question: User's question
        embeddings_model: The embedding model to use for similarity calculation
        max_context_tokens: Maximum tokens allowed for contexts
        
    Returns:
        List of truncated contexts that fit within token limit
    """
    if not contexts:
        return contexts
    
    # Calculate current context token count
    current_tokens = sum(count_tokens(ctx) for ctx in contexts)
    
    # If already within limit, return as-is (no unnecessary processing)
    if current_tokens <= max_context_tokens:
        logger.info(f"Context within limit: {current_tokens} tokens ≤ {max_context_tokens}, no truncation needed")
        return contexts
    
    logger.info(f"Context truncation needed: {current_tokens} tokens > {max_context_tokens} limit")
    
    try:
        # Get question embedding once for efficiency
        question_emb = embeddings_model.embed_query(question)
        
        # Score each context by multiple factors for better selection
        context_scores = []
        for i, ctx in enumerate(contexts):
            # Calculate similarity score
            ctx_emb = embeddings_model.embed_query(ctx)
            similarity = cosine_similarity([question_emb], [ctx_emb])[0][0]
            
            # Calculate tokens for this context
            tokens = count_tokens(ctx)
            
            # Calculate efficiency score (relevance per token)
            efficiency = similarity / max(tokens, 1)
            
            # Boost score for highly relevant contexts (similarity > 0.8)
            boosted_similarity = similarity * 1.2 if similarity > 0.8 else similarity
            
            # Combined score: balance between similarity and efficiency
            combined_score = (boosted_similarity * 0.7) + (efficiency * 0.3)
            
            context_scores.append({
                'index': i,
                'context': ctx,
                'similarity': similarity,
                'boosted_similarity': boosted_similarity,
                'tokens': tokens,
                'efficiency': efficiency,
                'combined_score': combined_score
            })
        
        # Sort by combined score (best contexts first)
        context_scores.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Strategy 1: Select whole contexts that fit within limit
        selected_contexts = []
        total_tokens = 0
        remaining_contexts = []
        
        for ctx_info in context_scores:
            if total_tokens + ctx_info['tokens'] <= max_context_tokens:
                selected_contexts.append(ctx_info)
                total_tokens += ctx_info['tokens']
                logger.info(f"Selected context {ctx_info['index']}: {ctx_info['tokens']} tokens, similarity: {ctx_info['similarity']:.3f}, combined_score: {ctx_info['combined_score']:.3f}")
            else:
                remaining_contexts.append(ctx_info)
        
        # Strategy 2: Try to fit partial content from highly relevant remaining contexts
        remaining_token_budget = max_context_tokens - total_tokens
        
        if remaining_token_budget > 200:  # Only if we have substantial space left
            for ctx_info in remaining_contexts:
                # Only consider highly relevant contexts for partial inclusion
                if ctx_info['similarity'] > 0.75 and remaining_token_budget > 200:
                    logger.info(f"Attempting partial truncation for highly relevant context {ctx_info['index']} (similarity: {ctx_info['similarity']:.3f})")
                    
                    truncated_ctx = truncate_context_smartly(
                        ctx_info['context'], 
                        remaining_token_budget - 50,  # Leave some buffer
                        question, 
                        embeddings_model
                    )
                    
                    if truncated_ctx and count_tokens(truncated_ctx) > 100:  # Only if meaningful content remains
                        truncated_tokens = count_tokens(truncated_ctx)
                        selected_contexts.append({
                            **ctx_info,
                            'context': truncated_ctx,
                            'tokens': truncated_tokens,
                            'truncated': True
                        })
                        total_tokens += truncated_tokens
                        remaining_token_budget -= truncated_tokens
                        logger.info(f"Partially included context {ctx_info['index']}: {truncated_tokens} tokens (truncated from {ctx_info['tokens']})")
                        
                        if remaining_token_budget < 200:  # Stop if we're running low on space
                            break
        
        # Sort selected contexts back to original order for coherence
        selected_contexts.sort(key=lambda x: x['index'])
        final_contexts = [ctx_info['context'] for ctx_info in selected_contexts]
        
        # Final validation
        actual_tokens = sum(count_tokens(ctx) for ctx in final_contexts)
        
        logger.info(
            f"Smart context truncation completed:\n"
            f"  Original: {len(contexts)} contexts, {current_tokens} tokens\n"
            f"  Final: {len(final_contexts)} contexts, {actual_tokens} tokens\n"
            f"  Reduction: {current_tokens - actual_tokens} tokens ({((current_tokens - actual_tokens) / current_tokens * 100):.1f}%)"
        )
        
        return final_contexts
        
    except Exception as e:
        logger.error(f"Error in smart context truncation: {e}")
        # Enhanced fallback: select contexts by similarity rather than just first ones
        try:
            question_emb = embeddings_model.embed_query(question)
            fallback_scores = []
            
            for i, ctx in enumerate(contexts):
                ctx_emb = embeddings_model.embed_query(ctx)
                similarity = cosine_similarity([question_emb], [ctx_emb])[0][0]
                tokens = count_tokens(ctx)
                fallback_scores.append((i, ctx, similarity, tokens))
            
            # Sort by similarity
            fallback_scores.sort(key=lambda x: x[2], reverse=True)
            
            selected = []
            total_tokens = 0
            for i, ctx, sim, tokens in fallback_scores:
                if total_tokens + tokens <= max_context_tokens:
                    selected.append(ctx)
                    total_tokens += tokens
                else:
                    break
            
            logger.info(f"Fallback truncation: selected {len(selected)} contexts with {total_tokens} tokens")
            return selected
            
        except:
            # Ultimate fallback: simple truncation
            selected = []
            total_tokens = 0
            for ctx in contexts:
                ctx_tokens = count_tokens(ctx)
                if total_tokens + ctx_tokens <= max_context_tokens:
                    selected.append(ctx)
                    total_tokens += ctx_tokens
                else:
                    break
            return selected


def truncate_context_smartly(context: str, max_tokens: int, question: str, embeddings_model) -> str:
    """
    Truncate a single context by keeping the most relevant sentences.
    Enhanced with better sentence boundary detection and relevance scoring.
    """
    if not context.strip():
        return ""
    
    # Try different sentence splitting approaches for better accuracy
    sentences = []
    
    # Method 1: Split on '. ' but be smarter about abbreviations
    potential_sentences = context.split('. ')
    for i, sent in enumerate(potential_sentences):
        sent = sent.strip()
        if not sent:
            continue
            
        # Add period back except for the last sentence
        if i < len(potential_sentences) - 1:
            sent += '.'
            
        sentences.append(sent)
    
    # If we only got one "sentence", try splitting on other punctuation
    if len(sentences) <= 1:
        # Try splitting on other sentence endings
        import re
        sentences = re.split(r'[.!?]+\s+', context.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
    
    # If still only one sentence or very few words, do word-based truncation
    if len(sentences) <= 1:
        words = context.split()
        # Estimate tokens per word (roughly 0.75 tokens per word for English)
        estimated_words = int(max_tokens * 1.3)  # Conservative estimate
        
        if len(words) <= estimated_words:
            return context
        
        # Truncate by words and add ellipsis
        truncated = ' '.join(words[:estimated_words])
        # Try to end at a reasonable point (not mid-sentence)
        if '.' in truncated[-50:]:  # Look for period in last 50 chars
            last_period = truncated.rfind('.')
            truncated = truncated[:last_period + 1]
        else:
            truncated += "..."
            
        return truncated
    
    try:
        # Get question embedding for sentence scoring
        question_emb = embeddings_model.embed_query(question)
        
        # Score sentences by relevance and other factors
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
            
            # Calculate similarity to question
            sent_emb = embeddings_model.embed_query(sentence)
            similarity = cosine_similarity([question_emb], [sent_emb])[0][0]
            
            # Calculate token count
            tokens = count_tokens(sentence + (' ' if not sentence.endswith('.') else ''))
            
            # Position bonus: slightly prefer earlier sentences (intro context)
            position_bonus = 0.1 * (1 - i / len(sentences)) if len(sentences) > 1 else 0
            
            # Length penalty: slightly penalize very short sentences (< 10 tokens)
            length_penalty = 0.05 if tokens < 10 else 0
            
            # Final score combining multiple factors
            final_score = similarity + position_bonus - length_penalty
            
            sentence_scores.append({
                'sentence': sentence,
                'similarity': similarity,
                'tokens': tokens,
                'final_score': final_score,
                'index': i
            })
        
        # Sort by final score (best sentences first)
        sentence_scores.sort(key=lambda x: x['final_score'], reverse=True)
        
        # Select sentences that fit within token limit
        selected_sentences = []
        total_tokens = 0
        
        for sent_info in sentence_scores:
            if total_tokens + sent_info['tokens'] <= max_tokens:
                selected_sentences.append(sent_info)
                total_tokens += sent_info['tokens']
            elif total_tokens == 0:  # If even the first sentence doesn't fit, truncate it
                # Truncate this sentence to fit
                words = sent_info['sentence'].split()
                estimated_words = int(max_tokens * 1.3)
                if len(words) > estimated_words:
                    truncated_sentence = ' '.join(words[:estimated_words]) + "..."
                    selected_sentences.append({
                        **sent_info,
                        'sentence': truncated_sentence,
                        'tokens': count_tokens(truncated_sentence)
                    })
                    total_tokens += count_tokens(truncated_sentence)
                break
        
        if not selected_sentences:
            return ""
        
        # Sort selected sentences back to original order for readability
        selected_sentences.sort(key=lambda x: x['index'])
        
        # Reconstruct text
        result_sentences = [s['sentence'] for s in selected_sentences]
        result = ' '.join(result_sentences)
        
        # Clean up the result
        if not result.endswith(('.', '!', '?')):
            result += '.'
            
        return result
        
    except Exception as e:
        logger.warning(f"Error in smart sentence truncation: {e}, falling back to word truncation")
        
        # Final fallback: simple word-based truncation
        if count_tokens(context) <= max_tokens:
            return context
        
        words = context.split()
        estimated_words = int(max_tokens * 1.3)  # Conservative estimation
        truncated = ' '.join(words[:estimated_words])
        
        # Try to end at a sentence boundary
        if '.' in truncated[-50:]:
            last_period = truncated.rfind('.')
            truncated = truncated[:last_period + 1]
        else:
            truncated += "..."
            
        return truncated

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