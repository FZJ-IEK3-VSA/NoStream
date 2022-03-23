FROM python:3.9-slim-buster

WORKDIR /app

COPY . .
COPY requirements.txt ./requirements.txt

# MILP solver for 
RUN apt-get update
RUN apt-get install -y coinor-cbc

RUN pip install --no-cache-dir -r requirements.txt

CMD streamlit run ./streamlit/streamlit_app.py