# Multi-stage build keeps the runtime image small.
FROM python:3.11-slim AS builder

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip build \
    && pip install --no-cache-dir .

# ---- runtime ----
FROM python:3.11-slim AS runtime

# Run as non-root for security.
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY src ./src
COPY data ./data
COPY scripts ./scripts

# Train the baseline at build time so the image ships ready-to-serve.
RUN python -m triageml.train_baseline && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "triageml.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
