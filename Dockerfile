FROM python:3.8.12-buster

COPY requirements.txt requirements.txt
COPY API API

RUN apt-get update  \
&&  apt-get --yes install libsndfile1 \
&& pip install --upgrade pip wheel \
&& pip install -r requirements.txt

CMD uvicorn API.Api:app --host 0.0.0.0
#  Add that before pushing to GCP :            --port $PORT
