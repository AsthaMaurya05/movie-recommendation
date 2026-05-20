import pandas as pd
import ast
import pickle
import os
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt', quiet=True)
ps = PorterStemmer()

# --- DATA LOADING ---
movies = pd.read_csv('data/tmdb_5000_movies.csv')
credits = pd.read_csv('data/tmdb_5000_credits.csv')

movies = movies.merge(credits, on='title')
movies = movies.drop_duplicates(subset='movie_id')
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# --- PARSING FUNCTIONS ---
def convert(text):
    return [item['name'] for item in ast.literal_eval(text)]

def convert_cast(text):
    return [item['name'] for i, item in enumerate(ast.literal_eval(text)) if i < 2]

def fetch_director(text):
    for item in ast.literal_eval(text):
        if item['job'] == 'Director':
            return [item['name']]
    return []

def collapse_spaces(lst):
    return [item.replace(" ", "") for item in lst]

# --- APPLY PARSING ---
movies['genres']   = movies['genres'].apply(convert).apply(collapse_spaces)
movies['keywords'] = movies['keywords'].apply(convert).apply(collapse_spaces)
movies['cast']     = movies['cast'].apply(convert_cast).apply(collapse_spaces)
movies['crew']     = movies['crew'].apply(fetch_director).apply(collapse_spaces)
movies['overview'] = movies['overview'].fillna('').apply(lambda x: x.split())

# --- BUILD TAGS (director weighted 3x) ---
movies['tags'] = (
    movies['overview'] +
    movies['genres']*2   +
    movies['keywords']*2 +
    movies['cast']     +
    movies['crew'] * 3
)
movies['tags'] = movies['tags'].apply(lambda x: " ".join(x).lower())

# --- STEMMING ---
def stem(text):
    return " ".join([ps.stem(word) for word in text.split()])

final = movies[['movie_id', 'title', 'tags']].copy()
final['tags'] = final['tags'].apply(stem)

# --- VECTORIZATION & SIMILARITY ---
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
vectors = tfidf.fit_transform(final['tags'])
similarity = cosine_similarity(vectors)

# --- SAVE ARTIFACTS ---
os.makedirs('artifacts', exist_ok=True)
pickle.dump(final, open('artifacts/movies.pkl', 'wb'))
pickle.dump(similarity, open('artifacts/similarity.pkl', 'wb'))
print("Artifacts saved!")

# --- RECOMMENDER (for testing) ---
def recommend(movie_title):
    movie_index = final[final['title'] == movie_title].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    print("\nBecause you liked " + movie_title + ", you might enjoy:\n")
    for i, (idx, score) in enumerate(movie_list, 1):
        print(str(i) + ". " + final.iloc[idx]['title'])

recommend("Inception")