import pandas as pd
import numpy as np
import ast
import nltk
import streamlit as st

from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download stopwords
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

def stem(text):
    return " ".join([ps.stem(word) for word in text.split()])


# ----------------------------
# LOAD DATA
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('tmdb_5000_movies.csv')

    df = df[['title', 'overview', 'genres', 'keywords']]
    df.dropna(inplace=True)

    df['genres'] = df['genres'].apply(convert)
    df['keywords'] = df['keywords'].apply(convert)

    df['overview'] = df['overview'].apply(lambda x: x.split())

    df['tags'] = df['overview'] + df['genres'] + df['keywords']
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
# STREAMLIT UI
# ----------------------------
st.title("🎬 Movie Recommendation System")

# ---- SHOW DATA (for demo) ----
st.subheader("📊 Explore Data")

if st.checkbox("Show processed dataset"):
    st.write(df.head())

if st.checkbox("Show tags (final NLP text)"):
    st.write(df[['title', 'tags']].head())

if st.checkbox("Show vectors"):
    st.write(vectors[:5])

if st.checkbox("Show similarity matrix"):
    st.write(similarity[:5, :5])


# ---- INPUT ----
movie_name = st.text_input("Enter a movie name")

# ---- RECOMMEND ----
if st.button("Recommend"):
    movie = movie_name.lower()

    if movie not in df['title'].str.lower().values:
        st.write("❌ Movie not found")
    else:
        index = df[df['title'].str.lower() == movie].index[0]
        distances = similarity[index]

        movies_list = sorted(
            list(enumerate(distances)),
            reverse=True,
            key=lambda x: x[1]
        )[1:6]

        st.write("### 🎯 Recommended Movies:")

        for i in movies_list:
            st.write(
                "👉",
                df.iloc[i[0]].title,
                "| Similarity Score:",
                round(i[1], 2)
            )
