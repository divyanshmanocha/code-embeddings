FROM python:3.7
COPY . /app
WORKDIR /app
RUN pip3 install -r requirements.txt
EXPOSE 8080
CMD streamlit run --server.port 8080 --server.enableCORS false app.py
