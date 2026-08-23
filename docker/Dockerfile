FROM python:3.11-slim

WORKDIR /app

# CPU-only torch/torchvision, installed separately from PyPI (which only hosts CUDA
# builds) and before the rest of requirements since this layer is large and changes rarely.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.13.0 torchvision==0.28.0

COPY docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY models/cnn_baseline.pt models/cnn_baseline.pt

ENV MODEL_PATH=models/cnn_baseline.pt

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
