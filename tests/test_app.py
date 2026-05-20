import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app

def test_home_page():
    client = app.test_client()
    response = client.get('/')
    assert response.status_code == 200

def test_valid_recommendation():
    client = app.test_client()
    response = client.post('/recommend',
                           json={'title': 'Inception'},
                           content_type='application/json')
    assert response.status_code == 200
    data = response.get_json()
    assert 'recommendations' in data
    assert len(data['recommendations']) == 5

def test_invalid_movie():
    client = app.test_client()
    response = client.post('/recommend',
                           json={'title': 'XYZABC123'},
                           content_type='application/json')
    assert response.status_code == 404