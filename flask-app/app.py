from flask import Flask,render_template,request,redirect
import pandas as pd
import numpy as np
import os
import io
import requests
import pickle

import torch
from transformers import DistilBertTokenizer, DistilBertModel


app_tv_show_recommender = Flask(__name__)

app_tv_show_recommender.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

app_tv_show_recommender.model = DistilBertModel.from_pretrained('distilbert-base-uncased', output_hidden_states = True)

app_tv_show_recommender.embeddings = np.load('CLS_matrix.npy')

with open('index_to_name_dict.pickle', 'rb') as f:
    app_tv_show_recommender.index_to_name_dict = pickle.load(f)

@app_tv_show_recommender.route('/',methods=['GET','POST'])
def index(): 
    '''
    This function sets up the main page and asks for an input sentence
    '''
    if request.method == 'GET':
        return render_template('welcome.html')
    else:
    	#request was a POST
    	if request.form['action'] == 'About':
    		return redirect('/about')
    	else:
            app_tv_show_recommender.user_entered_sentence = request.form['sentence_description']
            # user entered sentence for what they want to see

            app_tv_show_recommender.top_recommendations_list = recommend(app_tv_show_recommender.user_entered_sentence)

            #return render_template('recommendations.html', recommendations_list = top_recommendations_list)

            return redirect('/recommendations')

def tokenize(sentence):
  '''
  Returns an embedding for the sentence.
  Specifically, returns the output of the CLS token in the penultimate model layer
  '''
  inputs = app_tv_show_recommender.tokenizer(sentence, return_tensors="pt")
  outputs = app_tv_show_recommender.model(**inputs)
  penultimate = outputs[-1][-2]

  return penultimate[:, 0, :].squeeze().detach().numpy()

def recommend(user_sentence: str, n: int = 5):
    '''
    This function:
    1. takes in a sentence, and calculates its embedding based on the tokenizer and model being used
    2. calculates cosine similarity of the resulting embedding against the embeddings in our database
    3. returns the indices of the n closest vectors, sorted from the closest vector to the furthest away, in a list
    '''

    user_embedding = tokenize(user_sentence).astype('float16')
    cosine_similarity_vector = np.dot(app_tv_show_recommender.embeddings, user_embedding)

    # Next, we fetch the indices for the n largest entries in cosine_similarity_vector, in descending order
    top_indices = np.argpartition(cosine_similarity_vector, len(cosine_similarity_vector) - n)[-n:]
    top_array = np.vstack([top_indices, cosine_similarity_vector[top_indices]]).T

    '''
    top_array is an array of shape (n, 2) where the first column are the indices and the second column the corresponding cosine similarity values.
    All that remains to do is sort top_array in descending order according to the second column, and fetch the indices in the corresponding order.
    '''

    top_indices_sorted = top_array[top_array[:, 1].argsort()[::-1]][:,0].astype(int)

    return [app_tv_show_recommender.index_to_name_dict.get(index, '') for index in top_indices_sorted]

@app_tv_show_recommender.route('/recommendations',methods=['GET','POST'])
def display_recommendations():
    '''
    In this function we display the recommended TV shows
    '''
    
    if request.method == 'GET':
        return render_template('recommendations.html', recommendations_list = app_tv_show_recommender.top_recommendations_list)

    else:
        return redirect('/')

@app_tv_show_recommender.route('/about',methods=['GET','POST'])
def about(): 
    if request.method == 'GET':
        return render_template('about.html')
    else:
        return redirect('/')


#port = int(os.environ.get("PORT", 5000))
#app_tv_show_recommender.run(host='0.0.0.0', port='5001', debug=True)

if __name__ == "__main__":
    #args, unknown = parser.parse_known_args()
    port = int(os.environ.get("PORT", 5000))
    app_tv_show_recommender.run(host='0.0.0.0', port=port, debug=True)
