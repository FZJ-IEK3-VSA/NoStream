FROM python:3.9-slim-bullseye

WORKDIR /app


COPY requirements.txt .


# MILP solver for 
RUN apt-get update
RUN apt-get install -y coinor-cbc

RUN pip install -r requirements.txt

COPY index.html /usr/local/lib/python3.9/site-packages/streamlit/static/index.html

COPY . .

EXPOSE 8501
EXPOSE 8502

CMD streamlit run ./streamlit/streamlit_app.py --server.enableCORS false --server.enableXsrfProtection false --server.enableWebsocketCompression=false
