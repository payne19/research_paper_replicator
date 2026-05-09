import os
import torch
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from unstructured.partition.pdf import partition_pdf
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer
from scraper_arxiv_links import WebScraper
from ingestion_vector_db import load_from_user, get_user_store

load_dotenv("creds.env")

S2_API_KEY      = os.getenv("SCHOLAR_SEMANTIC")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL")
S2_HEADERS      = {"x-api-key": S2_API_KEY} if S2_API_KEY else {}
S2_BASE         = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS       = "title,authors,year,publicationDate,externalIds,citationCount,referenceCount,fieldsOfStudy,url"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title="Arxiv Research Replicator", layout="wide")
st.title("Arxiv Research Replicator")


def init_session():
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if "user_store" not in st.session_state:
        st.session_state.user_store = get_user_store(st.session_state.embeddings)

    if "loaded_doc" not in st.session_state:
        st.session_state.loaded_doc = None

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "prompt" not in st.session_state:
        with open("prompt.txt", "r") as f:
            st.session_state.prompt = f.read().strip()

    if "tokenizer" not in st.session_state:
        st.session_state.tokenizer = AutoTokenizer.from_pretrained(LOCAL_LLM_MODEL)

    if "model" not in st.session_state:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        st.session_state.model = AutoModelForCausalLM.from_pretrained(
            LOCAL_LLM_MODEL,
            quantization_config=quant_config,
            device_map="auto",
        )
        st.session_state.model.eval()


init_session()


def extract_arxiv_id(url):
    if "arxiv.org" in url:
        return url.rstrip("/").split("/")[-1].split("v")[0]
    return ""


def fetch_s2_metadata(arxiv_id):
    if not arxiv_id:
        return {}
    r = requests.get(
        f"{S2_BASE}/paper/arXiv:{arxiv_id}",
        params={"fields": S2_FIELDS},
        headers=S2_HEADERS,
    )
    return r.json() if r.status_code == 200 else {}


def build_s2_metadata(source_type, url="", arxiv_id="", filename=""):
    meta = {
        "source_type":   source_type,
        "source_url":    url,
        "filename":      filename,
        "arxiv_id":      arxiv_id,
        "source":        "user",
        "has_full_text": "true",
    }
    s2 = fetch_s2_metadata(arxiv_id)
    if s2:
        authors = [a.get("name", "") for a in s2.get("authors", [])]
        meta.update({
            "title":           s2.get("title", ""),
            "authors":         "|".join(authors),
            "author_count":    str(len(authors)),
            "year":            str(s2.get("year", "")),
            "published_at":    s2.get("publicationDate", ""),
            "citation_count":  str(s2.get("citationCount", 0)),
            "reference_count": str(s2.get("referenceCount", 0)),
            "fields_of_study": "|".join(s2.get("fieldsOfStudy", [])),
            "s2_paper_id":     s2.get("paperId", ""),
            "s2_url":          s2.get("url", ""),
        })
    return meta


def extract_pdf_text(path):
    elements = partition_pdf(
        filename=path,
        strategy="hi_res",
        extract_images_in_pdf=True,
        extract_image_block_types=["Image", "Table"],
        extract_image_block_to_payload=True,
    )
    return " ".join(el.text for el in elements if el.text).strip()


def generate_response(question, metadata):
    chunks = st.session_state.user_store.similarity_search(question, k=5)

    context = "\n\n".join(
        f"[Chunk {i+1}]\n{c.page_content}" for i, c in enumerate(chunks)
    )

    meta_str = "\n".join(f"{k}: {v}" for k, v in metadata.items() if v)

    full_prompt = (
        f"{st.session_state.prompt}\n\n"
        f"Paper Metadata:\n{meta_str}\n\n"
        f"Relevant Excerpts:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    inputs    = st.session_state.tokenizer(full_prompt, return_tensors="pt").to(DEVICE)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = st.session_state.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=st.session_state.tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][input_len:]
    return st.session_state.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


st.radio("Select a source:", ("Arxiv URL", "PDF Upload", "Text"), key="source_selection")

if st.session_state.source_selection == "Arxiv URL":
    url = st.text_input("Enter the Arxiv URL:")
    if st.button("Scrape and Process"):
        if url:
            with st.spinner("Scraping and storing..."):
                full_text = WebScraper(url).scrape_arxiv().strip().replace("\n", " ")
                arxiv_id  = extract_arxiv_id(url)
                metadata  = build_s2_metadata("url", url=url, arxiv_id=arxiv_id)
                load_from_user(st.session_state.user_store, full_text, metadata)
                st.session_state.loaded_doc = metadata
                st.session_state.messages   = []
            st.success(f"Stored. S2 metadata: {'fetched' if metadata.get('s2_paper_id') else 'not found'}")
        else:
            st.error("Please enter a valid URL.")

elif st.session_state.source_selection == "PDF Upload":
    uploaded_file = st.file_uploader("Upload a PDF file:")
    url           = st.text_input("Arxiv or DOI URL to fetch metadata (optional):")
    if st.button("Process PDF"):
        if uploaded_file:
            temp_path = f"temp_{uploaded_file.name}"
            try:
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                with st.spinner("Extracting and storing..."):
                    full_text = extract_pdf_text(temp_path)
                    arxiv_id  = extract_arxiv_id(url) if url else ""
                    metadata  = build_s2_metadata("pdf", url=url, arxiv_id=arxiv_id, filename=uploaded_file.name)
                    load_from_user(st.session_state.user_store, full_text, metadata)
                    st.session_state.loaded_doc = metadata
                    st.session_state.messages   = []
                st.success("PDF stored.")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        else:
            st.error("Please upload a PDF file.")

elif st.session_state.source_selection == "Text":
    text_input = st.text_area("Paste your research text:")
    url        = st.text_input("Arxiv or DOI URL to fetch metadata (optional):")
    if st.button("Process Text"):
        if text_input.strip():
            full_text = text_input.strip().replace("\n", " ")
            arxiv_id  = extract_arxiv_id(url) if url else ""
            metadata  = build_s2_metadata("text", url=url, arxiv_id=arxiv_id)
            load_from_user(st.session_state.user_store, full_text, metadata)
            st.session_state.loaded_doc = metadata
            st.session_state.messages   = []
            st.success("Text stored.")
        else:
            st.error("Please enter some text.")

st.divider()

if st.session_state.loaded_doc:
    doc_meta = st.session_state.loaded_doc
    st.caption(f"Active paper: **{doc_meta.get('title') or doc_meta.get('filename') or doc_meta.get('source_url', 'Unknown')}**")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if question := st.chat_input("Ask anything about this paper..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = generate_response(question, doc_meta)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()
else:
    st.info("Upload a paper above to start chatting.")

if st.button("Reload creds.env"):
    load_dotenv("creds.env", override=True)
    st.success("Reloaded.")