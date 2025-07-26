FROM --platform=linux/amd64 python:3.10-slim

WORKDIR /app
COPY app/ /app/

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "process_pdfs.py"]
