FROM python:3.10-slim

WORKDIR /app


RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY download_models.py .


COPY . .


CMD ["uvicorn", "caption_server:app", "--host", "0.0.0.0", "--port", "8000"]
