#Imports and basic config
import os, json, uuid, re, math, time, gc, glob
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

import fitz
import pdfplumber
import pandas as pd
from unidecode import unidecode
from rapidfuzz import fuzz
from rank_bm25 import BM25Okapi

import numpy as np
import faiss


# Embeddings (small, multilingual)
from sentence_transformers import SentenceTransformer

import tabula

# USER INPUTS
# 1) Mount Google Drive manually in Colab (Runtime > Connect to Drive) or upload PDFs to /content/pdfs
PDF_DIR = "content/index"  # change to your folder path if different
INDEX_DIR = "content/index"  # where we will save FAISS and metadata
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# 2) API keys (read from Hugging Face secrets / env variables)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
AGROMONITORING_API_KEY = os.getenv("AGROMONITORING_API_KEY")
DATA_GOV_IN_API_KEY = os.getenv("DATA_GOV_IN_API_KEY")


# Chunking/Embedding constants
TEXT_CHUNK_TOKENS = 450
TEXT_CHUNK_OVERLAP = 64
ROW_KEY_DIMENSIONS = ["crop","growth_stage","soil_type","region"]
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # small, multilingual

# Retrieval configuration
RETRIEVE_K_DENSE = 8
RETRIEVE_K_SPARSE = 8
MERGE_K = 5  # cap after merge
RERANK_TOP = 3 # we will implement remote reranker later with Groq

#Defining schema dataclasses
@dataclass
class QualityFlags:
    extracted_ok: bool
    table_parse_conf: float
    staleness_days: Optional[int] = None

@dataclass
class Citations:
    page_no: Optional[int] = None
    table_id: Optional[str] = None
    row_key: Optional[str] = None

@dataclass
class ChunkMetadata:
    doc_id: str
    title: str
    publisher: Optional[str]
    date: Optional[str]  # "YYYY-MM-DD" if known
    doc_type: Optional[str]
    domains: List[str]
    crops: List[str]
    regions: List[str]
    growth_stage: Optional[str]
    language: str  # "en" | "hi" | mixed
    page_no: Optional[int]
    section_heading: Optional[str]
    table_id: Optional[str]
    table_caption: Optional[str]
    units_system: str
    quality_flags: QualityFlags
    citations: Citations

@dataclass
class Chunk:
    id: str
    chunk_type: str  # "text" | "table_schema" | "table_row"
    text: str
    bm25_text: str
    metadata: ChunkMetadata
    # embeddings vector stored separately

def norm_text(s: str) -> str:
    # Normalize whitespace and convert smart quotes, remove excessive spaces
    s = s.replace("\xa0"," ").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def detect_language(text: str) -> str:
    # Simple heuristic: presence of Devanagari unicode range → hi; else en/mixed
    if re.search(r"[\u0900-\u097F]", text):
        return "hi"
    return "en"

# Basic synonym maps
CROP_SYNONYMS = {
    "chana": ["bengal gram","gram","चना"],
    "paddy": ["rice","धान","चावल"],
    "wheat": ["गेहूं","gehu","gehun"],
    "maize": ["corn","मक्का","bhutta","makka"],
    "mustard": ["sarson","सरसों"],
    "okra": ["bhindi","भिंडी"],
    "cotton": ["कपास","kapas"]
}

REGION_SYNONYMS = {
    "uttar pradesh": ["up","उत्तर प्रदेश"],
    "madhya pradesh": ["mp","मध्य प्रदेश"],
    "bihar": ["बिहार"],
    "maharashtra": ["mh","महाराष्ट्र"],
    "tamil nadu": ["tn","तमिलनाडु"]
}

def normalize_units(cell: str) -> str:
    # Simple pass-through; expand to convert e.g., l/ha, kg/ha normalization
    if cell is None:
        return ""
    return norm_text(cell)

def build_row_key(row: Dict[str, Any]) -> str:
    parts = []
    for k in ROW_KEY_DIMENSIONS:
        v = row.get(k, "")
        parts.append(f"{k}={str(v).lower()}")
    return "|".join(parts)

def sanitize_table_df(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure string col names
    df.columns = [str(c).strip() if c is not None else "" for c in df.columns]
    # make unique names if repeated
    seen = {}
    new_cols = []
    for c in df.columns:
        if c not in seen:
            seen[c] = 0
            new_cols.append(c)
        else:
            seen[c] += 1
            new_cols.append(f"{c}_{seen[c]}")
    df.columns = new_cols
    # Drop columns that are entirely empty or NaN
    df = df.dropna(axis=1, how="all")
    # Drop fully empty rows
    df = df.dropna(axis=0, how="all")
    # Fill remaining NaN with empty strings for robust string ops
    df = df.fillna("")
    # Convert everything to str for consistent downstream parsing
    for c in df.columns:
        df[c] = df[c].astype(str).map(lambda x: x.strip())
    return df

#Extract text chunks from PDF with section/page awareness
def chunk_text(text: str, max_tokens=TEXT_CHUNK_TOKENS, overlap=TEXT_CHUNK_OVERLAP) -> List[str]:
    # Token proxy by words
    words = text.split()
    chunks = []
    i = 0
    step = max_tokens - overlap
    while i < len(words):
        chunk_words = words[i:i+max_tokens]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        i += step
    return chunks

def extract_text_chunks(pdf_path: str, doc_meta_base: Dict[str, Any]) -> List[Chunk]:
    doc = fitz.open(pdf_path)
    chunks = []
    title_guess = os.path.basename(pdf_path)
    for page_no in range(len(doc)):
        page = doc.load_page(page_no)
        text = page.get_text("text")
        text = norm_text(text)
        if not text:
            continue
        lang = detect_language(text)
        # naive heading detection: first line or lines in caps
        heading = None
        lines = text.split("\n")
        if lines:
            cand = lines[0].strip()
            if len(cand) > 0 and (cand.isupper() or len(cand) < 80):
                heading = cand
        # chunk
        for piece in chunk_text(text):
            chunk_id = str(uuid.uuid4())
            md = ChunkMetadata(
                doc_id=doc_meta_base["doc_id"],
                title=doc_meta_base.get("title", title_guess),
                publisher=doc_meta_base.get("publisher"),
                date=doc_meta_base.get("date"),
                doc_type=doc_meta_base.get("doc_type", "guideline"),
                domains=doc_meta_base.get("domains", []),
                crops=doc_meta_base.get("crops", []),
                regions=doc_meta_base.get("regions", []),
                growth_stage=None,
                language=lang,
                page_no=page_no+1,
                section_heading=heading,
                table_id=None,
                table_caption=None,
                units_system="metric",
                quality_flags=QualityFlags(True, 1.0, None),
                citations=Citations(page_no=page_no+1, table_id=None, row_key=None),
            )
            chunks.append(Chunk(
                id=chunk_id,
                chunk_type="text",
                text=piece,
                bm25_text=piece,
                metadata=md
            ))
    doc.close()
    return chunks

#Extract tables with pdfplumber/tabula
def extract_tables(pdf_path: str, doc_meta_base: Dict[str, Any]) -> List[Chunk]:
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            try:
                tables = page.extract_tables()
            except Exception:
                tables = []
            for t_idx, table in enumerate(tables or []):
                df = pd.DataFrame(table[1:], columns=table[0])
                df = df.dropna(how="all")
                table_id = f"{os.path.basename(pdf_path)}_p{i+1}_t{t_idx+1}"
                cols = []
                for c in df.columns:
                    cols.append({"name": str(c), "unit": None, "description": None})
                schema_text = "Table columns: " + ", ".join([c["name"] for c in cols])
                chunk_id = str(uuid.uuid4())
                md = ChunkMetadata(
                    doc_id=doc_meta_base["doc_id"],
                    title=doc_meta_base.get("title", os.path.basename(pdf_path)),
                    publisher=doc_meta_base.get("publisher"),
                    date=doc_meta_base.get("date"),
                    doc_type=doc_meta_base.get("doc_type", "guideline"),
                    domains=doc_meta_base.get("domains", []),
                    crops=doc_meta_base.get("crops", []),
                    regions=doc_meta_base.get("regions", []),
                    growth_stage=None,
                    language="en",
                    page_no=i+1,
                    section_heading=None,
                    table_id=table_id,
                    table_caption=None,
                    units_system="metric",
                    quality_flags=QualityFlags(True, 0.9, None),
                    citations=Citations(page_no=i+1, table_id=table_id, row_key=None),
                )
                chunks.append(Chunk(
                    id=chunk_id,
                    chunk_type="table_schema",
                    text=schema_text,
                    bm25_text=schema_text,
                    metadata=md
                ))
                # row chunks
                for _, row in df.iterrows():
                    row_dict = {}
                    for c in df.columns:
                        row_dict[str(c).strip().lower()] = normalize_units(str(row[c]) if pd.notna(row[c]) else "")
                    row_key = build_row_key({
                        "crop": row_dict.get("crop",""),
                        "growth_stage": row_dict.get("stage",""),
                        "soil_type": row_dict.get("soil",""),
                        "region": row_dict.get("region","")
                    })
                    # render compact textual row
                    parts = [f"{k}: {v}" for k,v in row_dict.items()]
                    row_text = f"Row ({row_key}) | " + " | ".join(parts)
                    chunk_id = str(uuid.uuid4())
                    md_row = md
                    md_row = ChunkMetadata(**asdict(md))
                    md_row.citations = Citations(page_no=i+1, table_id=table_id, row_key=row_key)
                    chunks.append(Chunk(
                        id=chunk_id,
                        chunk_type="table_row",
                        text=row_text,
                        bm25_text=row_text,
                        metadata=md_row
                    ))
    return chunks

def extract_tables(pdf_path: str, doc_meta_base: Dict[str, Any]) -> List[Chunk]:
    chunks = []
    base_title = doc_meta_base.get("title", os.path.basename(pdf_path))

    def add_table_chunks(df: pd.DataFrame, page_num: int, t_idx: int):
        nonlocal chunks
        if df is None or df.empty:
            return
        df = sanitize_table_df(df)
        if df.empty:
            return
        table_id = f"{os.path.basename(pdf_path)}_p{page_num}_t{t_idx}"
        # Schema chunk
        cols = [{"name": c, "unit": None, "description": None} for c in df.columns]
        schema_text = "Table columns: " + ", ".join([c["name"] for c in cols])
        md_schema = ChunkMetadata(
            doc_id=doc_meta_base["doc_id"],
            title=base_title,
            publisher=doc_meta_base.get("publisher"),
            date=doc_meta_base.get("date"),
            doc_type=doc_meta_base.get("doc_type", "guideline"),
            domains=doc_meta_base.get("domains", []),
            crops=doc_meta_base.get("crops", []),
            regions=doc_meta_base.get("regions", []),
            growth_stage=None,
            language="en",
            page_no=page_num,
            section_heading=None,
            table_id=table_id,
            table_caption=None,
            units_system="metric",
            quality_flags=QualityFlags(True, 0.9, None),
            citations=Citations(page_no=page_num, table_id=table_id, row_key=None),
        )
        chunks.append(Chunk(
            id=str(uuid.uuid4()),
            chunk_type="table_schema",
            text=schema_text,
            bm25_text=schema_text,
            metadata=md_schema
        ))
        # Row chunks
        for _, row in df.iterrows():
            row_dict = {str(c).strip().lower(): normalize_units(str(row[c])) for c in df.columns}
            row_key = build_row_key({
                "crop": row_dict.get("crop",""),
                "growth_stage": row_dict.get("stage",""),
                "soil_type": row_dict.get("soil",""),
                "region": row_dict.get("region","")
            })
            row_text = f"Row ({row_key}) | " + " | ".join([f"{k}: {v}" for k,v in row_dict.items()])
            md_row = ChunkMetadata(
                doc_id=md_schema.doc_id,
                title=md_schema.title,
                publisher=md_schema.publisher,
                date=md_schema.date,
                doc_type=md_schema.doc_type,
                domains=md_schema.domains,
                crops=md_schema.crops,
                regions=md_schema.regions,
                growth_stage=None,
                language=md_schema.language,
                page_no=page_num,
                section_heading=None,
                table_id=table_id,
                table_caption=None,
                units_system="metric",
                quality_flags=QualityFlags(True, 0.9, None),
                citations=Citations(page_no=page_num, table_id=table_id, row_key=row_key),
            )
            chunks.append(Chunk(
                id=str(uuid.uuid4()),
                chunk_type="table_row",
                text=row_text,
                bm25_text=row_text,
                metadata=md_row
            ))

    # Pass 1: pdfplumber
    pages_had_tables = False
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            try:
                tables = page.extract_tables()
            except Exception:
                tables = []
            if not tables:
                continue
            pages_had_tables = True
            for t_idx, table in enumerate(tables, start=1):
                try:
                    if not table or len(table) < 2:
                        continue
                    header = table[0]
                    rows = table[1:]
                    # Replace None or empty strings; generate synthetic headers if needed
                    if not header or all([(h is None or str(h).strip()=="") for h in header]):
                        ncols = max(len(r) for r in rows) if rows else 0
                        header = [f"col_{j+1}" for j in range(ncols)]
                    # Ensure row lengths match header
                    clean_rows = []
                    for r in rows:
                        if len(r) < len(header):
                            r = list(r) + [""]*(len(header)-len(r))
                        elif len(r) > len(header):
                            r = r[:len(header)]
                        clean_rows.append(r)
                    df = pd.DataFrame(clean_rows, columns=header)
                    df = sanitize_table_df(df)
                    if not df.empty:
                        add_table_chunks(df, page_num=i+1, t_idx=t_idx)
                except Exception as e:
                    pass # Continue without killing the whole PDF

    # Pass 2: Tabula fallback if no tables found via pdfplumber
    if not pages_had_tables:
        try:
            # Try lattice first (works for ruled tables)
            dfs = tabula.read_pdf(pdf_path, pages="all", lattice=True, multiple_tables=True)
            if not dfs:
                # Try stream for whitespace-separated tables
                dfs = tabula.read_pdf(pdf_path, pages="all", stream=True, multiple_tables=True)
            if dfs:
                for idx, df in enumerate(dfs, start=1):
                    try:
                        df = sanitize_table_df(df)
                        if not df.empty:
                            add_table_chunks(df, page_num=idx, t_idx=idx)
                    except Exception:
                        pass
        except Exception:
            pass

    return chunks

#Build hybrid index classes
class HybridIndex:
    def __init__(self, embedding_model_name=EMBEDDING_MODEL_NAME):
        self.model = SentenceTransformer(embedding_model_name)
        self.faiss_index = None
        self.embeddings = None
        self.bm25 = None
        self.corpus_texts = []
        self.chunks: List[Chunk] = []

    def add_chunks(self, new_chunks: List[Chunk]):
        self.chunks.extend(new_chunks)

    def build(self):
        # Prepare texts
        self.corpus_texts = [c.text for c in self.chunks]
        # Sparse
        tokenized = [t.split() for t in self.corpus_texts]
        self.bm25 = BM25Okapi(tokenized)
        # Dense
        emb = self.model.encode(self.corpus_texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
        self.embeddings = emb.astype("float32")
        d = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(d)  # inner product for normalized vectors
        self.faiss_index.add(self.embeddings)

    def save(self, index_dir=INDEX_DIR):
        os.makedirs(index_dir, exist_ok=True)
        # Save FAISS
        faiss.write_index(self.faiss_index, os.path.join(index_dir, "faiss.index"))
        # Save embeddings
        np.save(os.path.join(index_dir, "embeddings.npy"), self.embeddings)
        # Save corpus + metadata
        with open(os.path.join(index_dir, "corpus_texts.json"), "w") as f:
            json.dump(self.corpus_texts, f)
        with open(os.path.join(index_dir, "chunks.jsonl"), "w") as f:
            for c in self.chunks:
                f.write(json.dumps({
                    "id": c.id,
                    "chunk_type": c.chunk_type,
                    "text": c.text,
                    "bm25_text": c.bm25_text,
                    "metadata": asdict(c.metadata)
                }) + "\n")

    def load(self, index_dir=INDEX_DIR):
        # Load FAISS
        self.faiss_index = faiss.read_index(os.path.join(index_dir, "faiss.index"))
        self.embeddings = np.load(os.path.join(index_dir, "embeddings.npy"))
        with open(os.path.join(index_dir, "corpus_texts.json"), "r") as f:
            self.corpus_texts = json.load(f)
        self.chunks = []
        with open(os.path.join(index_dir, "chunks.jsonl"), "r") as f:
            for line in f:
                obj = json.loads(line)
                md = obj["metadata"]
                chunk = Chunk(
                    id=obj["id"],
                    chunk_type=obj["chunk_type"],
                    text=obj["text"],
                    bm25_text=obj["bm25_text"],
                    metadata=ChunkMetadata(
                        doc_id=md["doc_id"],
                        title=md["title"],
                        publisher=md.get("publisher"),
                        date=md.get("date"),
                        doc_type=md.get("doc_type"),
                        domains=md.get("domains", []),
                        crops=md.get("crops", []),
                        regions=md.get("regions", []),
                        growth_stage=md.get("growth_stage"),
                        language=md.get("language","en"),
                        page_no=md.get("page_no"),
                        section_heading=md.get("section_heading"),
                        table_id=md.get("table_id"),
                        table_caption=md.get("table_caption"),
                        units_system=md.get("units_system","metric"),
                        quality_flags=QualityFlags(**md.get("quality_flags", {"extracted_ok":True,"table_parse_conf":1.0,"staleness_days":None})),
                        citations=Citations(**md.get("citations", {}))
                    )
                )
                self.chunks.append(chunk)
        # Rebuild BM25
        tokenized = [t.split() for t in self.corpus_texts]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, k_dense=RETRIEVE_K_DENSE, k_sparse=RETRIEVE_K_SPARSE, merge_k=MERGE_K):
        # Sparse
        sparse_scores = self.bm25.get_scores(query.split())
        top_sparse_idx = np.argsort(sparse_scores)[::-1][:k_sparse]
        # Dense
        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        D, I = self.faiss_index.search(q_emb, k_dense)
        top_dense_idx = I[0]
        # Merge by normalized score
        scores = {}
        # normalize sparse
        if len(top_sparse_idx) > 0:
            smax = sparse_scores[top_sparse_idx[0]] if sparse_scores[top_sparse_idx[0]] > 0 else 1.0
        else:
            smax = 1.0
        for idx in top_sparse_idx:
            scores[idx] = scores.get(idx, 0.0) + (sparse_scores[idx]/smax)
        # normalize dense
        for rank, idx in enumerate(top_dense_idx):
            sim = D[0][rank]
            scores[idx] = scores.get(idx, 0.0) + float(sim)
        # sort and cap
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:merge_k]
        results = [(self.chunks[i], s) for i,s in ranked]
        return results

#Create a simple manifest by scanning PDF_DIR
def default_doc_meta(pdf_path: str) -> Dict[str, Any]:
    base = os.path.basename(pdf_path)
    # crude guesses
    return {
        "doc_id": str(uuid.uuid4()),
        "title": os.path.splitext(base)[0],
        "publisher": None,
        "date": None,
        "doc_type": "guideline",
        "domains": ["agronomy","irrigation","pest"],  # broad for now
        "crops": [],  # leave empty; retrieval will be semantic
        "regions": [],
    }

pdf_files = sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))

# Run ingestion and build index
# hybrid = HybridIndex(embedding_model_name=EMBEDDING_MODEL_NAME)

# total_chunks = 0
# for pdf_path in pdf_files:
#     meta = default_doc_meta(pdf_path)
#     try:
#         t_chunks = extract_text_chunks(pdf_path, meta)
#         tb_chunks = extract_tables(pdf_path, meta)
#         hybrid.add_chunks(t_chunks + tb_chunks)
#         total_chunks += len(t_chunks) + len(tb_chunks)
#         print(f"Ingested {os.path.basename(pdf_path)}: text={len(t_chunks)}, table_chunks={len(tb_chunks)}")
#     except Exception as e:
#         print("Failed:", pdf_path, e)

# print("Total chunks:", total_chunks)
# hybrid.build()
# hybrid.save(INDEX_DIR)
# print("Index built and saved to", INDEX_DIR)
# hybrid = HybridIndex(embedding_model_name=EMBEDDING_MODEL_NAME)

# Test search
hybrid2 = HybridIndex(embedding_model_name=EMBEDDING_MODEL_NAME)
hybrid2.load(INDEX_DIR)

def preview_hit(hit, score):
    c = hit
    print(f"[{c.chunk_type}] score={score:.3f} page={c.metadata.page_no} table_id={c.metadata.table_id}")
    print(c.text[:300], "...")
    print("Doc:", c.metadata.title)
    print("—")


"""# **Phase 2**"""

from typing import Tuple
import time, hashlib, json
import requests
from datetime import datetime
import re

# Simple intent classifier
def classify_intent(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ["irrigation", "paani", "water"]):
        return "irrigation"
    if any(w in q for w in ["pest", "keeda", "bollworm", "disease", "threshold", "etl"]):
        return "pest_disease"
    if any(w in q for w in ["price", "daam", "mandi", "rate", "bazaar"]):
        return "market_price"
    if any(w in q for w in ["policy", "yojana", "scheme", "kcc", "pm-kisan"]):
        return "policy"
    if any(w in q for w in ["weather", "mausam", "rain", "temperature", "baarish"]):
        return "weather"
    return "general"

# Query expansion based on intent
def expand_query(query: str, intent: str) -> str:
    q = query
    if intent == "irrigation":
        q += " irrigation schedule water requirement mm per irrigation frequency critical stages guideline"
    elif intent == "pest_disease":
        q += " economic threshold level ETL control measures official agri advisory"
    elif intent == "market_price":
        q += " mandi daily price market rates arrivals agmarknet"
    return q.strip()

def retrieve_with_confidence(query: str, hybrid_idx: HybridIndex,
                              k_dense=RETRIEVE_K_DENSE, k_sparse=RETRIEVE_K_SPARSE) -> Tuple[list, float, str]:
    # Classify and expand query
    intent = classify_intent(query)
    expanded_q = expand_query(query, intent)

    # Search hybrid index
    results = hybrid_idx.search(expanded_q, k_dense=k_dense, k_sparse=k_sparse)

    boosted_results = []
    for chunk, score in results:
        boost = 0.0
        # Prefer table rows for agronomy/pest
        if chunk.chunk_type == "table_row":
            boost += 0.4
        # Boost based on domain match
        if intent == "irrigation" and "irrigation" in (chunk.metadata.domains or []):
            boost += 0.3
        if intent == "pest_disease" and any(d in chunk.metadata.domains for d in ["pest","disease"]):
            boost += 0.3
        # Down-weight irrelevant docs
        if any(bad in (chunk.metadata.title or "").lower()
               for bad in ["insurance","pmfby","market intelligence","scheme","policy"]):
            boost -= 0.5

        boosted_results.append((chunk, score + boost))

    boosted_results = sorted(boosted_results, key=lambda x: x[1], reverse=True)

    # Confidence = top score normalized against arbitrary scale
    top_score = boosted_results[0][1] if boosted_results else 0
    confidence = min(1.0, max(0.0, top_score / 2.0))  # scale to 0-1

    return boosted_results[:RERANK_TOP], confidence, intent


CACHE = {}
CACHE_TTL = {
    "weather": 600,       # 10 minutes
    "soil": 10800,        # 3 hours
    "ndvi": 86400,        # 24 hours
    "market": 86400,      # 24 hours
    "web": 21600          # 6 hours
}

def make_cache_key(name: str, params: dict) -> str:
    raw = name + json.dumps(params, sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()

def cache_get(name, params):
    key = make_cache_key(name, params)
    if key in CACHE:
        item = CACHE[key]
        if time.time() - item["time"] < CACHE_TTL.get(name, 3600):
            return item["data"]
    return None

def cache_set(name, params, data):
    key = make_cache_key(name, params)
    CACHE[key] = {"data": data, "time": time.time()}

def get_weather(lat, lon):
    params = {"lat": lat, "lon": lon, "appid": os.environ["OPENWEATHER_API_KEY"], "units": "metric"}
    cached = cache_get("weather", params)
    if cached:
        return cached

    url = "https://api.openweathermap.org/data/2.5/onecall"
    try:
        r = requests.get(url, params=params, timeout=5)
        r.raise_for_status()
        data = r.json()
        cache_set("weather", params, data)
        return data
    except Exception as e:
        return {"error": str(e)}

def get_soil_and_ndvi(polygon_id):
    soil_params = {"polyid": polygon_id, "appid": os.environ["AGROMONITORING_API_KEY"]}
    cached_soil = cache_get("soil", soil_params)
    if not cached_soil:
        try:
            soil_url = f"http://api.agromonitoring.com/agro/1.0/soil"
            r = requests.get(soil_url, params=soil_params, timeout=5)
            r.raise_for_status()
            cached_soil = r.json()
            cache_set("soil", soil_params, cached_soil)
        except Exception as e:
            cached_soil = {"error": str(e)}

    ndvi_params = {"polyid": polygon_id, "appid": os.environ["AGROMONITORING_API_KEY"]}
    cached_ndvi = cache_get("ndvi", ndvi_params)
    if not cached_ndvi:
        try:
            ndvi_url = f"http://api.agromonitoring.com/agro/1.0/ndvi/history"
            r = requests.get(ndvi_url, params=ndvi_params, timeout=5)
            r.raise_for_status()
            cached_ndvi = r.json()
            cache_set("ndvi", ndvi_params, cached_ndvi)
        except Exception as e:
            cached_ndvi = {"error": str(e)}

    return {"soil": cached_soil, "ndvi": cached_ndvi}

# helper to extract state/district/market from query
STATE_LIST = [
    "Uttar Pradesh", "Madhya Pradesh", "West Bengal", "Bihar", "Maharashtra",
    "Punjab", "Haryana", "Rajasthan", "Gujarat", "Karnataka", "Andhra Pradesh",
    "Tamil Nadu", "Odisha", "Telangana", "Chhattisgarh", "Jharkhand"
]

def extract_location_from_query(query):
    found_state = None
    for s in STATE_LIST:
        if s.lower() in query.lower():
            found_state = s
            break
    return found_state  # we can extend later for district/market

def get_market_price(commodity, state=None, district=None, market=None):
    resource_id = "35985678-0d79-46b4-9ed6-6f13308a1d24"
    params = {
        "api-key": os.environ["DATA_GOV_IN_API_KEY"],
        "format": "json",
        "limit": 3,
        "filters[commodity]": commodity
    }
    if state:
        params["filters[state]"] = state
    if district:
        params["filters[district]"] = district
    if market:
        params["filters[market]"] = market

    cached = cache_get("market", params)
    if cached:
        return cached

    try:
        url = f"https://api.data.gov.in/resource/35985678-0d79-46b4-9ed6-6f13308a1d24"
        r = requests.get(url, params=params, timeout=3)
        r.raise_for_status()
        data = r.json()
        cache_set("market", params, data)
        return data
    except Exception as e:
        return {"error": str(e)}


def tavily_search(query):
    params = {
        "query": query + " site:gov.in OR site:nic.in OR site:imd.gov.in OR site:agmarknet.gov.in OR site:enam.gov.in",
        "api_key": os.environ["TAVILY_API_KEY"]
    }
    cached = cache_get("web", params)
    if cached:
        return cached

    url = "https://api.tavily.com/search"
    try:
        r = requests.post(url, json=params, timeout=6)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and "results" in data and isinstance(data["results"], list):
            data["results"] = data["results"][:3]
        cache_set("web", params, data)
        return data
    except Exception as e:
        return {"error": str(e)}


def router(query):
    hits, conf, intent = retrieve_with_confidence(query, hybrid2)

    # High confidence KB answer
    if conf >= 1.99 and intent not in ["market_price", "weather"]:
        return {
            "source": "KB",
            "answer": hits[0][0].text,
            "confidence": conf
        }

    # Market price handler: Call Agmarknet live API
    if intent == "market_price":
        words = query.lower().split()
        commodity = None
        state = None
        for w in words:
            if w in ["wheat", "chana", "rice", "paddy", "mustard", "cotton", "okra"]:
                commodity = w.capitalize()
        if not commodity:
            commodity = "Wheat"

        price_data = get_market_price(commodity, state=state)

        if "records" in price_data and price_data["records"]:
            lines = []
            for rec in price_data["records"]:
                lines.append(f"{rec['market']} ({rec['district']}): ₹{rec['modal_price']} on {rec['arrival_date']}")
            return {
                "source": "Market API (Agmarknet)",
                "answer": "\n".join(lines),
                "confidence": conf
            }
        else:
            return {
                "source": "Market API (Agmarknet)",
                "answer": "No recent price data available for your query.",
                "confidence": conf
            }

    # Weather handler
    if intent == "weather":
        weather_data = get_weather(28.6139, 77.2090)
        return {"source": "Weather API", "answer": weather_data, "confidence": conf}

    # For irrigation/pest questions:
    if intent in ["pest_disease", "irrigation"]:
        if conf >= 1.99:
            return {"source": "KB", "answer": hits[0][0].text, "confidence": conf}
        else:
            web_data = tavily_search(query)
            return {"source": "Web (Tavily)", "answer": web_data, "confidence": conf}

    # Fallback for general queries
    web_data = tavily_search(query)
    return {"source": "Web (Tavily)", "answer": web_data, "confidence": conf}


def get_weather_simple(city_name):
    params = {"q": city_name, "appid": os.environ["OPENWEATHER_API_KEY"], "units": "metric"}
    url_current = "https://api.openweathermap.org/data/2.5/weather"
    try:
        r = requests.get(url_current, params=params, timeout=5)
        r.raise_for_status()
        data = r.json()
        return {
            "temp": data["main"]["temp"],
            "description": data["weather"][0]["description"],
            "humidity": data["main"]["humidity"],
            "wind": data["wind"]["speed"]
        }
    except Exception as e:
        return {"error": str(e)}

def router(query):
    hits, conf, intent = retrieve_with_confidence(query, hybrid2)

    # High confidence KB
    if conf >= 0.99 and intent not in ["market_price", "weather"]:
        return {
            "source": "KB",
            "answer": hits[0][0].text,
            "confidence": conf
        }

    # MARKET PRICE
    if intent == "market_price":
        state = extract_location_from_query(query)
        commodity = None
        for w in query.lower().split():
            if w in ["wheat", "chana", "rice", "paddy", "mustard", "cotton", "okra"]:
                commodity = w.capitalize()
        if not commodity:
            commodity = "Wheat"

        price_data = get_market_price(commodity, state=state)

        if "records" in price_data and price_data["records"]:
            header = f"Commodity: {commodity} | State: {state or 'All India'}"
            lines = [header]
            for rec in price_data["records"]:
                lines.append(f"{rec['market']} ({rec['district']}): ₹{rec['modal_price']} on {rec['arrival_date']}")
            return {
                "source": "Market API (Agmarknet)",
                "answer": "\n".join(lines),
                "confidence": conf
            }
        else:
            return {
                "source": "Market API (Agmarknet)",
                "answer": "No recent price data available for your query.",
                "confidence": conf
            }

    # WEATHER
    if intent == "weather":
        # TEMPORARY: use city simple API
        city = query.split()[-1]  # naive parse
        weather_data = get_weather_simple(city)
        return {"source": "Weather API", "answer": weather_data, "confidence": conf}

    # PEST/IRRIGATION
    if intent in ["pest_disease", "irrigation"]:
        if conf >= 0.5:
            return {"source": "KB", "answer": hits[0][0].text, "confidence": conf}
        else:
            web_data = tavily_search(query)
            return {"source": "Web (Tavily)", "answer": web_data, "confidence": conf}

    # Fallback
    web_data = tavily_search(query)
    return {"source": "Web (Tavily)", "answer": web_data, "confidence": conf}

"""# **Phase 3**"""

from groq import Groq
import os

# Set your key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=os.environ["GROQ_API_KEY"])

tools_def = [
    {
        "type": "function",
        "function": {
            "name": "get_market_price",
            "description": "Fetch mandi price (Agmarknet) for a commodity",
            "parameters": {
                "type": "object",
                "properties": {
                    "commodity": {"type": "string", "description": "e.g., Wheat, Chana, Paddy"},
                    "state": {"type": "string"},
                    "district": {"type": "string"},
                    "market": {"type": "string"}
                },
                "required": ["commodity"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current/forecast weather; pass a city name or 'lat,lon'",
            "parameters": {
                "type": "object",
                "properties": {
                    "city_or_latlon": {"type": "string", "description": "e.g., Kolkata or '22.57,88.36'"}
                },
                "required": ["city_or_latlon"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tavily_search",
            "description": "Search official government sources for policies or advisories",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    }
]

def groq_stream_final_answer(messages, model="openai/gpt-oss-120b", temperature=0.2):
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=True
    )
    final_text = []
    for chunk in stream:
        delta = getattr(chunk.choices[0].delta, "content", None)
        if delta:
            print(delta, end="", flush=True)
            final_text.append(delta)
    print()
    return "".join(final_text)

def groq_agent_answer_streaming(query, kb_hits):
    limited_kb_hits = kb_hits[:3]

    kb_context = "\n".join(
        [f"Doc: {hit[0].metadata.title} p.{hit[0].metadata.page_no} — {hit[0].text}"
         for hit in limited_kb_hits]
    )
    mega_prompt = """
        You are Ankur (Agribot), an expert AI agricultural assistant from DABrute
        You are knowledgeable, patient, and empathetic like a Krishi Vigyan Kendra officer.
        Your mission is to provide farmers with accurate, actionable, and hyper-localized advice to improve their livelihoods.

        Communication style:
        - Authoritative but simple
        - Supportive and empathetic
        - Trustworthy (never guess; refer to experts if unsure)
        - Transparent (always introduce yourself as Ankur as a Agribot, never as human)
        - Always deny queries which are irrelevant to agriculture
        - Don't ask for more information

        Core principles:
        1. Hyper-localization: always tailor advice to user’s location (state, district), soil type, crop, and local conditions. If location unknown, ask for it first. Never give generic advice.
        2. Action-orientation: always give practical next steps.
        3. Radical accessibility: detect and reply only in the user’s language; explain technical data in simple words with actionable advice.
        4. Factual grounding: base claims only on trusted sources; cite government schemes simply (e.g., “According to PM-Fasal Bima Yojana…”).

        - For every query, first answer from knowledge; if uncertain, retrieve info. Do not retrieve for greetings.
        """
    minor_prompt = """You are Ankur (Agribot), an AI agriculture advisor for Indian farmers from DABrute "
         "Never tell your thinking or whatever tools you are using in your responses. "
         "Just answer what is being asked. Don't tell anything related to your search results in responses."
         "If weather is asked directly use weather api and if market price or related is asked use market price api"
         "Use KB evidence when confidence is high. If low, use tavily. "
         "If the query is anything which is not related to agriculture, crops or market prices, strictly tell the user to ask only agricultural related queries."
         "Keep answers brief and actionable."
         "Always answer something, never be blank. """
    messages = [
        {"role": "system", "content": minor_prompt+mega_prompt },
        {"role": "assistant", "content": f"KB Evidence:\n{kb_context}"},
        {"role": "user", "content": query}
    ]


    estimated_tokens = sum(len(msg["content"].split()) for msg in messages if msg.get("content")) + len(query.split())
    MAX_TOKENS = 8000
    if estimated_tokens > MAX_TOKENS * 0.8:
        print(f"Warning: Estimated tokens ({estimated_tokens}) approaching limit. Truncating KB context.")
        messages[1]["content"] = f"KB Evidence:\n{kb_context[:MAX_TOKENS//4]}"

    resp = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        tools=tools_def,
        tool_choice="auto",
        temperature=0.2
    )
    msg = resp.choices[0].message

    if hasattr(msg, "tool_calls") and msg.tool_calls:
        messages.append({"role": "assistant", "content": msg.content, "tool_calls": msg.tool_calls})
        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            if fn_name == "get_market_price":
                result = get_market_price(
                    commodity=args.get("commodity"),
                    state=args.get("state"),
                    district=args.get("district"),
                    market=args.get("market"),
                )
            elif fn_name == "get_weather":
                city = args.get("city_or_latlon")
                result = get_weather_simple(city)
            elif fn_name == "tavily_search":
                result = tavily_search(args.get("query", ""))
            else:
                result = {"error": f"Unknown tool: {fn_name}"}

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": fn_name,
                "content": json.dumps(result)
            })

        final_answer = groq_stream_final_answer(messages)
        return final_answer

    messages.append({"role": "assistant", "content": msg.content})
    final_answer = groq_stream_final_answer(messages)
    return final_answer

#Testing
def test_groq_agent():
    tests = [
        # "Irrigation schedule for wheat during flowering stage",
        # "Watering tips for pea plants",
        # "Current mandi price for wheat in Uttar Pradesh",
        # "Weather forecast for Kolkata tomorrow",
        # "Latest crop insurance policy for PMFBY"
        "who is president of india?",
        # "Which crops should i sow in winter?",
        "Based on current weather in nagpur, what should be the best crop for profit?"
    ]
    for q in tests:
        print("="*100)
        hits, conf, intent = retrieve_with_confidence(q, hybrid2)
        print(f"User Query: {q}\n[DEBUG] Intent: {intent} | Confidence: {conf:.2f}\n")
        ans = groq_agent_answer_streaming(q, hits)
        # print("\n— end of answer —\n")

#test_groq_agent()

def test_groq_agent():
    tests = [
        "price of bhindi in kharagpur"
    ]
    for q in tests:
        print("="*100)
        hits, conf, intent = retrieve_with_confidence(q, hybrid2)
        print(f"User Query: {q}\n[DEBUG] Intent: {intent} | Confidence: {conf:.2f}\n")
        ans = groq_agent_answer_streaming(q, hits)

        # print("\n— end of answer —\n")

#test_groq_agent()

"""# **Phase 4 : User Interfacce**"""

import gradio as gr
import speech_recognition as sr
from gtts import gTTS
import tempfile
from typing import List, Tuple
import os
import time

class AgriBot:
    """
    Encapsulates the logic for the AgriBot, integrating the backend functions
    from the notebook with the Gradio UI.
    """
    def __init__(self):
        self.recognizer = sr.Recognizer()

    def process_text_query(self, message: str, history: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], str]:
        """
        Processes a text query by retrieving context from the knowledge base (KB)
        and generating a response using the Groq agent.
        """
        history = history or []

        try:
            hits, confidence, intent = retrieve_with_confidence(message, hybrid2)

            history.append((message, None))

            response = groq_agent_answer_streaming(message, hits)

            history[-1] = (message, response)

            return history, ""
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            history.append((message, error_msg))
            return history, ""

    def text_to_speech(self, text: str) -> str:
        """Converts text to an MP3 audio file using gTTS."""
        if not text:
            return None
        try:
            tts = gTTS(text=text, lang='en', slow=False)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_fp:
                tts.save(temp_fp.name)
                return temp_fp.name
        except Exception as e:
            print(f"TTS Error: {e}")
            return None

    def speech_to_text(self, audio_filepath) -> str:
        """Converts an audio file to text using Google's Speech Recognition."""
        if audio_filepath is None:
            return "No audio recorded. Please try again."
        try:
            with sr.AudioFile(audio_filepath) as source:
                audio_data = self.recognizer.record(source)
                # Recognize speech using Google Web Speech API
                text = self.recognizer.recognize_google(audio_data)
                return text
        except sr.UnknownValueError:
            return "Could not understand the audio. Please speak clearly."
        except sr.RequestError as e:
            return f"Speech recognition service unavailable; {e}"
        except Exception as e:
            return f"An error occurred while processing audio: {e}"

# Initialize a single instance of the bot
agribot = AgriBot()

"""# **Testing**

#ANKUR UI
"""

import gradio as gr
from gtts import gTTS
import speech_recognition as sr
import os
import uuid
import time

green_theme = gr.themes.Base(
    primary_hue=gr.themes.colors.green,
    secondary_hue=gr.themes.colors.lime,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_600",
)


def text_to_speech(text, lang='en'):
    """Converts text to an audio file."""
    if not text or not isinstance(text, str):
        print("Error: Invalid text provided for TTS.")
        return None
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        os.makedirs("content/audio_outputs", exist_ok=True)
        audio_filename = f"content/audio_outputs/{uuid.uuid4()}.mp3"
        tts.save(audio_filename)
        return audio_filename
    except Exception as e:
        print(f"Error during TTS conversion: {e}")
        return None

def speech_to_text(audio_filepath):
    """Transcribes speech from an audio file."""
    if not audio_filepath or not os.path.exists(audio_filepath):
        return ""

    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_filepath) as source:
            audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        return "Sorry, I could not understand the audio. Please try again."
    except sr.RequestError as e:
        return f"API Error: Could not request results; {e}"
    except Exception as e:
        print(f"An unexpected error in speech-to-text: {e}")
        return ""



chat_state = {
    "history": [],
    "last_response_audio": None
}


def ankur_chat_interface(user_text_input, user_audio_input):
    if user_audio_input:
        user_query = speech_to_text(user_audio_input)
    else:
        user_query = user_text_input
    if not user_query:
        return chat_state["history"], chat_state["last_response_audio"], ""


    hits, confidence, intent = retrieve_with_confidence(user_query, hybrid2)
    bot_response_text = groq_agent_answer_streaming(user_query, hits)

    bot_audio_response_path = text_to_speech(bot_response_text)
    chat_state["history"].append((user_query, bot_response_text))
    chat_state["last_response_audio"] = bot_audio_response_path

    return chat_state["history"], bot_audio_response_path, ""


def speak_last_response():
    return chat_state["last_response_audio"]



with gr.Blocks(theme=green_theme, css="""footer {display: none !important}
    #speak-btn {
    height: 200px !important;   /* custom button height */
    font-size: 14px;
    }
    #audio-player {
    height: 200px !important;   /* make audio player compact */
    width: 80%;
    font-size: 14px;
    
    }""", 
    title="Ankur The AgriBot") as demo:
    gr.Markdown("<div style='text-align: center; font-size: 34px;'> Ankur The AgriBot 🌱</div>")
    gr.Markdown("<div style='text-align: center; font-size: 12px;'>Ask me about weather updates, crop prices, or farming tips in real-time!</div>")

    chatbot_display = gr.Chatbot(label="Chat with AgriBot", height=450, bubble_full_width=False)

    with gr.Row():
        text_input_box = gr.Textbox(placeholder="Type your farming question here...", show_label=False, container=False, scale=5)
        send_button = gr.Button("Send", variant="primary", scale=1, min_width=150)

    with gr.Accordion("🎤 Record Your Voice Question", open=False):
        voice_input_audio = gr.Audio(sources=["microphone"], type="filepath", label="Click the record button and speak your question clearly.")
        process_voice_button = gr.Button("Process Voice Input")

    with gr.Row():
        speak_last_button = gr.Button("🔊 Speak Last Response", elem_id="speak-btn")
        response_audio_player = gr.Audio(label="Bot Response Audio", type="filepath", interactive=False, autoplay=False, elem_id="audio-player")


    text_submit_handler = send_button.click(fn=ankur_chat_interface, inputs=[text_input_box, gr.State(None)], outputs=[chatbot_display, response_audio_player, text_input_box])
    text_input_box.submit(fn=ankur_chat_interface, inputs=[text_input_box, gr.State(None)], outputs=[chatbot_display, response_audio_player, text_input_box])
    process_voice_button.click(fn=ankur_chat_interface, inputs=[gr.State(None), voice_input_audio], outputs=[chatbot_display, response_audio_player, text_input_box])
    speak_last_button.click(fn=speak_last_response, inputs=[], outputs=[response_audio_player])


demo.launch(debug=True)

