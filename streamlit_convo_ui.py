import streamlit as st
from scraper_playwright import WebScraper
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import psycopg2
import os  
from dotenv import load_dotenv

load_dotenv('creds.env')

st.title("Arxiv Research Replicator")

if 'embeddings' not in st.session_state:
    st.session_state['embeddings'] = HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDING_MODEL"))

st.radio("Select a source for your research replication:", ("Arxiv URL", "PDF Upload", "Text"), key="source_selection")

if st.session_state.source_selection == "Arxiv URL":
    url = st.text_input("Enter the Arxiv URL:")
    if st.button("Scrape and Process"):
        if url:
            scraper = WebScraper()
            scraped_text = scraper.scrape_arxiv(url)
            st.session_state['embeddings'].embed_documents([scraped_text])
            st.success("Scraping and embedding completed!")
        else:
            st.error("Please enter a valid URL.")

elif st.session_state.source_selection == "PDF Upload":
    uploaded_file = st.file_uploader("Upload a PDF file:")
    if st.button("Process PDF"):
        if uploaded_file:
            # Save the uploaded file to a temporary location
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        else:
            st.error("Please upload a PDF file.")

elif st.session_state.source_selection == "Text":
    text_input = st.text_area("Enter your research text:")
    if st.button("Process Text"):
        if text_input:
            st.session_state['embeddings'].embed_documents([text_input])
            st.success("Text embedding completed!")
        else:
            st.error("Please enter some text to process.")