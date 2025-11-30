FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# System deps for pypandoc (optional) and fonts for docx rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 pandoc \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py /app/
COPY mistral_ocr.py /app/

EXPOSE 8010
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8010", "--timeout-keep-alive", "75"]
