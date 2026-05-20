FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY recommender.py .
COPY templates/ templates/
COPY data/ data/

RUN python recommender.py

EXPOSE 5000

CMD ["python", "app.py"]