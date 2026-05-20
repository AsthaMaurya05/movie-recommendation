# -*- coding: utf-8 -*-
# UI implementation
from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load pre-computed artifacts (instant — no reprocessing)
movies     = pickle.load(open('artifacts/movies.pkl', 'rb'))
similarity = pickle.load(open('artifacts/similarity.pkl', 'rb'))

# Get sorted movie list for the search dropdown
movie_list = sorted(movies['title'].tolist())


def recommend(movie_title):
    """Returns top 5 similar movies"""
    idx       = movies[movies['title'] == movie_title].index[0]
    distances = similarity[idx]
    top5      = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    return [{"title": movies.iloc[i]['title'], "score": round(float(s), 3)} for i, s in top5]


@app.route('/')
def home():
    """Serve the main page, pass movie list to HTML"""
    return render_template('index.html', movies=movie_list)


@app.route('/recommend', methods=['POST'])
def get_recommendations():
    """API endpoint — receives movie title, returns JSON recommendations"""
    data  = request.get_json()
    title = data.get('title', '')

    if title not in movies['title'].values:
        return jsonify({'error': 'Movie not found'}), 404

    results = recommend(title)
    return jsonify({'recommendations': results})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)