import streamlit as st
import re
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

# Set page title and background color
st.set_page_config(page_title="Cinematic Subtitle Finder", page_icon="🎬", layout="centered", initial_sidebar_state="expanded")

# Title with stylized text and color
st.title("🎥 Cinematic Subtitle Finder 🍿")
st.markdown("---")

# Initialize ChromaDB client
client = chromadb.PersistentClient(path=r"C:\Users\hp\Desktop\abc")
client.heartbeat()

# Initialize Sentence Transformer embedding function
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Get collection from ChromaDB
collection = client.get_collection(name="Search_Engine", embedding_function=sentence_transformer_ef)

# Initialize Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to encode content
def encoding_content(x):
    return model.encode(x, normalize_embeddings=True)

# Function to clean text
def clean_text(text):
    text = re.sub(r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\r\n', '', text)
    text = re.sub(r'\r\n', ' ', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    text = re.sub(r'watch any video online with opensubtitles free browser extension osdblinkext', '', text)
    text = text.strip()
    return text

# Function to get search results
def get_results(query_text):
    query_clean = clean_text(query_text)
    query_em = encoding_content(query_clean)
    search_results = collection.query(query_embeddings=query_em.tolist(), n_results=10)
    return search_results

# Text input box for user query
user_search = st.text_input("Enter subtitle here...")
search_results =   get_results(user_search)

# Button to trigger search
if st.button("Search"):
    st.markdown("---")
    st.write("🔍 Top 10 Results:")
    for i, res in enumerate(search_results['metadatas'][0]):
        st.write(f"Result {i+1}: {res['title']}")

