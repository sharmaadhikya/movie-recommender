import pandas as pd
import numpy as np
import ast
import re
import nltk
import streamlit as st

from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')

ps = PorterStemmer()

# ----------------------------
# TEXT PROCESSING FUNCTIONS
# ----------------------------
def convert(text):
    try:
        return [i['name'] for i in ast.literal_eval(text)]
    except:
        return []

def get_director(text):
    try:
        for i in ast.literal_eval(text):
            if i['job'] == 'Director':
                return [i['name']]
    except:
        return []

def stem(text):
    return " ".join([ps.stem(word) for word in text.split()])


# ----------------------------
# LOAD DATA (CACHE FOR SPEED)
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('tmdb_5000_movies.csv')
    df = df[['title','overview','genres','keywords','cast','crew']]
    df.dropna(inplace=True)

    df['genres'] = df['genres'].apply(convert)
    df['keywords'] = df['keywords'].apply(convert)
    df['cast'] = df['cast'].apply(lambda x: convert(x)[:3])
    df['director'] = df['crew'].apply(get_director)

    df['overview'] = df['overview'].apply(lambda x: x.split())

    df['tags'] = df['overview'] + df['genres'] + df['keywords'] + df['cast'] + df['director']
    df['tags'] = df['tags'].apply(lambda x: " ".join(x).lower())
    df['tags'] = df['tags'].apply(stem)

    return df


df = load_data()

# ----------------------------
# VECTORIZE
# ----------------------------
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(df['tags']).toarray()

similarity = cosine_similarity(vectors)

# ----------------------------
# RECOMMEND FUNCTION
# ----------------------------
def recommend(movie):
    movie = movie.lower()
    
    if movie not in df['title'].str.lower().values:
        return []
    
    index = df[df['title'].str.lower() == movie].index[0]
    
    distances = similarity[index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    return [df.iloc[i[0]].title for i in movies_list]


# ----------------------------
# STREAMLIT UI
# ----------------------------
st.title("🎬 Movie Recommendation System")

movie_name = st.text_input("Enter a movie name")

if st.button("Recommend"):
    results = recommend(movie_name)

    if len(results) == 0:
        st.write("❌ Movie not found")
    else:
        st.write("### Recommended Movies:")
        for movie in results:
            st.write("👉", movie)
