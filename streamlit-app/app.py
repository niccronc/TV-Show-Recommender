import pandas as pd
import numpy as np
import pickle
import streamlit as st
import os
from transformers import DistilBertTokenizer, DistilBertModel

script_dir = os.path.dirname(os.path.abspath(__file__))
model_name = 'distilbert-base-uncased'
embedding_matrix_file_name = 'CLS_matrix.npy'
index_to_name_file_name = 'index_to_name_dict.pickle'

@st.cache_resource
def load_tokenizer():
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    return tokenizer

@st.cache_resource
def load_model():
    model = DistilBertModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval()
    return model

@st.cache_resource
def load_embeddings():
    embeddings = np.load(os.path.join(script_dir, embedding_matrix_file_name))
    with open(os.path.join(script_dir, index_to_name_file_name), 'rb') as f:
        index_to_name_dict = pickle.load(f)
    return embeddings, index_to_name_dict

st.title("TV Show Recommender")

tokenizer = load_tokenizer()
model = load_model()
embeddings, index_to_name_dict = load_embeddings()

st.divider()
with st.form("user_input"):
    num_recommendations = st.slider(
        label='How many recommendations do you want?',
        min_value=1,
        max_value=10,
        value=1,
        key='num_recommendations'
    )
    sentence = st.text_area(
        label="Describe the TV show you'd like to watch",
        key='user_sentence'
    )
    recommendation_submitted = st.form_submit_button()

@st.cache_data
def create_embedding(sentence: str):
    '''
    Returns an embedding for the sentence.
    Specifically, returns the output of the CLS token in the penultimate model layer
    '''
    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = model(**inputs)
    penultimate = outputs[-1][-2]

    return penultimate[:, 0, :].squeeze().detach().numpy()

if recommendation_submitted:

    user_embedding = create_embedding(sentence)
    cosine_similarity_vector = np.dot(embeddings, user_embedding) / np.linalg.norm(user_embedding)

    # We fetch the indices for the num_recommendations largest entries in cosine_similarity_vector, in descending order
    top_indices = np.argpartition(
        cosine_similarity_vector,
        len(cosine_similarity_vector) - num_recommendations
    )[-num_recommendations:]

    data = {
        'TV show': [index_to_name_dict.get(index, '') for index in top_indices],
        'cosine similarity': cosine_similarity_vector[top_indices].tolist()
    }
    df = pd.DataFrame(data).sort_values(by='cosine similarity', ascending=False)
    st.dataframe(df, hide_index=True)
