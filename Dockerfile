FROM node:22-alpine AS web-build
WORKDIR /build/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LLM_PROVIDER=ollama \
    LLM_MODEL=qwen2.5:3b \
    LLM_BASE_URL=http://host.docker.internal:11434 \
    RETRIEVAL_BACKEND=hybrid_ollama \
    EMBEDDING_MODEL=hf.co/CompendiumLabs/bge-base-en-v1.5-gguf:latest
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY --from=web-build /build/frontend/dist ./frontend/dist
RUN pip install --no-cache-dir .
RUN useradd --create-home --uid 10001 signalprep && chown -R signalprep:signalprep /app
USER signalprep
EXPOSE 8000
CMD ["uvicorn", "interview_coach.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
