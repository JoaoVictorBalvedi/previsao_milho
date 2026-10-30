# Interface web das previsoes (app/) com todos os modelos (sklearn, XGBoost, LightGBM, Chronos).
# Use pelo docker compose (sobe junto o PostgreSQL): docker compose up --build
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    HF_HOME=/cache/huggingface

# libgomp: OpenMP do XGBoost/LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /projeto
COPY requirements.txt .
# torch so de CPU (a versao padrao traz CUDA e ocupa varios GB a mais)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
