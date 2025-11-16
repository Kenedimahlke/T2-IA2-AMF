FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY automind-mcp/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY automind-mcp /app

EXPOSE 8000

CMD ["uvicorn", "server.http_server:app", "--host", "0.0.0.0", "--port", "8000"]
