import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

vectorstore = FAISS.load_local(
    "atlas_ya_desturi",
    embeddings,
    allow_dangerous_deserialization=True
)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

SOURCE_LABELS = {
    "african_authored":            "AFRICAN COMMUNITY VOICE",
    "kenyan_cultural_institution": "KENYAN CULTURAL INSTITUTION",
    "kenyan_academic":             "KENYAN ACADEMIC SOURCE",
    "western_academic":            "WESTERN ACADEMIC SOURCE",
    "unknown":                     "GENERAL SOURCE"
}

def is_useful_chunk(text):
    skip_phrases = [
        "bibliography", "references", "index", "table of contents",
        "isbn", "copyright", "all rights reserved", "published by",
        "doi:", "http", "www.", "et al."
    ]
    text_lower = text.lower()
    if sum(1 for p in skip_phrases if p in text_lower) > 2:
        return False
    if len(text.strip()) < 100:
        return False
    return True

def has_enough_context(context, min_chars=300):
    return context and len(context.strip()) > min_chars

def retrieve_by_source(query, k=20):
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": 50}
    )
    docs = retriever.invoke(query)
    docs = [d for d in docs if is_useful_chunk(d.page_content)]
    grouped = {}
    for doc in docs:
        source_type = doc.metadata.get("source_type", "unknown")
        if source_type not in grouped:
            grouped[source_type] = []
        grouped[source_type].append(doc)
    return grouped

def build_context_from_group(docs, max_chars=2000):
    context = ""
    for doc in docs:
        if len(context) >= max_chars:
            break
        context += doc.page_content + "\n\n"
    return context[:max_chars]

class Question(BaseModel):
    question: str

@app.post("/ask")
async def ask(body: Question):
    query = body.question
    grouped = retrieve_by_source(query)

    if not grouped:
        return {"answer": "Atlas Ya Desturi does not have verified information on this topic yet."}

    community_types = ["african_authored", "kenyan_cultural_institution"]
    community_docs, academic_docs = [], []
    community_source_names, academic_source_names = [], []

    for source_type, docs in grouped.items():
        label = SOURCE_LABELS.get(source_type, source_type)
        titles = list(set([d.metadata.get("title", "Unknown") for d in docs]))
        if source_type in community_types:
            community_docs.extend(docs)
            for t in titles:
                if t not in community_source_names:
                    community_source_names.append(f"{label}: {t}")
        else:
            academic_docs.extend(docs)
            for t in titles:
                if t not in academic_source_names:
                    academic_source_names.append(f"{label}: {t}")

    community_context = build_context_from_group(community_docs) if community_docs else None
    academic_context  = build_context_from_group(academic_docs)  if academic_docs  else None

    if not has_enough_context(community_context) and not has_enough_context(academic_context):
        return {"answer": "Atlas Ya Desturi does not have enough verified information on this topic yet."}

    if has_enough_context(community_context) and has_enough_context(academic_context):
        prompt = f"""You are Atlas Ya Desturi, an African cultural education assistant.

STRICT RULES:
- Only use information explicitly stated in the sources below
- Do NOT invent, assume or fill gaps with outside knowledge
- If the sources do not directly address the question, say so

ALL SOURCE CONTENT:
{academic_context}

{community_context}

QUESTION: {query}

Respond in EXACTLY this format:

FACTS:
- [verified fact directly from sources]
- [verified fact directly from sources]
- [verified fact directly from sources]

COMMUNITY VOICE:
[2-4 sentences of meaning, significance or lived experience from the community source.]

POSSIBLE BIASES:
[Only if there is a genuine framing difference. Otherwise write: No significant framing differences detected for this topic.]"""

    else:
        context = academic_context if has_enough_context(academic_context) else community_context
        source_type_label = "academic sources only" if has_enough_context(academic_context) else "community sources only"

        prompt = f"""You are Atlas Ya Desturi, an African cultural education assistant.

STRICT RULES:
- Only use information explicitly stated in the source below
- Do NOT invent, assume or fill gaps with outside knowledge

SOURCE CONTENT:
{context}

QUESTION: {query}

Respond in EXACTLY this format:

FACTS:
- [verified fact directly from source]
- [verified fact directly from source]
- [verified fact directly from source]

COMMUNITY VOICE:
[2-4 sentences of meaning or cultural context. If academic source only, note that community voice will be added as more sources are included.]"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=700
    )

    answer = response.choices[0].message.content
    all_sources = academic_source_names + community_source_names

    return {
        "answer": answer,
        "sources": all_sources
    }

@app.get("/")
def root():
    return {"status": "Atlas Ya Desturi API is running"}

