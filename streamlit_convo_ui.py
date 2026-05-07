import streamlit as st
from scraper_arxiv_links import WebScraper
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import psycopg2
import os  
from dotenv import load_dotenv
import tempfile
from unstructured.partition.pdf import partition_pdf

load_dotenv('creds.env')

st.title("Arxiv Research Replicator")

if 'embeddings' not in st.session_state:
    st.session_state['embeddings'] = HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDING_MODEL"))

def pdf_text_extractor(pdf_path):
    data = partition_pdf(
        filename=pdf_path,                  
        strategy="hi_res",                                     
        extract_images_in_pdf=True,                          
        extract_image_block_types=["Image", "Table"],         
        extract_image_block_to_payload=True,                
    )
    extracted_text = ''
    for i in data:
        extracted_text += i.text + '\n'
    return extracted_text

# def embedding_of_text(text):

st.radio("Select a source for your research replication:", ("Arxiv URL", "PDF Upload", "Text"), key="source_selection")

if st.session_state.source_selection == "Arxiv URL":
    url = st.text_input("Enter the Arxiv URL:")
    if st.button("Scrape and Process"):
        if url:
            scraper = WebScraper(url)
            scraped_text = scraper.scrape_arxiv()
            scraped_text = scraped_text.strip()
            scraped_text = scraped_text.replace('\n', ' ')
            scraped_text = scraped_text.lower()
            st.success("Scraping and embedding completed!")
        else:
            st.error("Please enter a valid URL.")

elif st.session_state.source_selection == "PDF Upload":
    uploaded_file = st.file_uploader("Upload a PDF file:")
    if st.button("Process PDF"):
        if uploaded_file:
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            extracted_text = pdf_text_extractor(temp_path)
            extracted_text = extracted_text.strip()
            extracted_text = extracted_text.replace('\n', ' ')
            extracted_text = extracted_text.lower()
            st.success("PDF processing and embedding completed!")
        else:
            st.error("Please upload a PDF file.")

elif st.session_state.source_selection == "Text":
    text_input = st.text_area("Enter your research text:")
    text_input = text_input.strip()
    text_input = text_input.replace('\n', ' ')
    text_input = text_input.lower()
    if st.button("Process Text"):
        if text_input:
            st.success("Text embedding completed!")
        else:
            st.error("Please enter some text to process.")

