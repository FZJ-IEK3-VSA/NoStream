FROM continuumio/miniconda3

EXPOSE $PORT
WORKDIR /app

COPY . .
COPY environment.yml ./environment.yml

RUN conda install python=3.9
RUN conda install -c conda-forge --file environment.yml

CMD streamlit run --server.port $PORT streamlit_app.py