import streamlit as st
import open_clip
import torch
import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import torch

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
model = SentenceTransformer('clip-ViT-B-32').to(device)

EMBEDDINGS_DIR = 'clip-vit-B32-features-org' 
IMAGES_DIR = 'Keyframes/keyframes'  


def load_embeddings():
    embeddings = []
    image_filenames = []
    for npy_file in os.listdir(EMBEDDINGS_DIR):
        if npy_file.endswith('.npy'):
            emb = np.load(os.path.join(EMBEDDINGS_DIR, npy_file))
            st.write(emb.shape)
            embeddings.append(emb)
            image_filenames.append(npy_file.replace('.npy', '.jpg'))  
    return np.array(embeddings), image_filenames

def get_text_embedding(query_text):
    text_embedding = model.encode([query_text], convert_to_tensor = False, normalize_embeddings=True)
    return np.array(text_embedding)

def search(embeddings, text_embedding, top_k=5):
    index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    distances, indices = index.search(text_embedding, top_k)
    return indices[0], distances[0]


embeddings, image_filenames = load_embeddings()

st.title("Image Search using OpenCLIP and Faiss")

query_text = st.text_input("Enter a query sentence", "")

if query_text:
    st.write(f"Searching for: {query_text}")
    text_embedding = get_text_embedding(query_text)
    indices, distances = search(embeddings, text_embedding)

    st.write("Top results:")
    for idx, distance in zip(indices, distances):
        image_path = os.path.join(IMAGES_DIR, image_filenames[idx])
        st.image(image_path, caption=f"{image_filenames[idx]} (Distance: {distance:.4f})", use_column_width=True)
