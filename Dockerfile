FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Initialise DB and RAG on startup, then launch Streamlit
CMD ["sh", "-c", "python -m data.seed_data && python -m rag.embeddings && streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true"]
