# Bangla Dialect Detection (RegSpeech12) — Dockerized Inference API

This project predicts Bangla regional dialects from speech audio. It packages a trained dialect classifier behind a small **FastAPI** service and ships a **Docker + docker-compose** setup so anyone can run the same environment and reproduce inference results.

The repository includes:
- a ready-to-run container (system audio deps + pinned Python requirements),
- exported model artifacts under `artifacts/runs/latest/`,
- evaluation reports under `artifacts/runs/reports/`,
- training / experimentation notebook(s).

---

## Dataset

Trained and evaluated using the **RegSpeech12** dataset (Kaggle):
- https://www.kaggle.com/datasets/mdrezuwanhassan/regspeech12/data

> Note: The dataset audio is **not** included in this repository.

---

## Preferred dialects (recommended for use)

Based on test-set results, this model is most reliable for:

1. Rangpur  
2. Chittagong  
3. Sylhet  
4. Barishal  
5. Kishoreganj  
6. Narail  
7. Sandwip  

(Other dialects may be predicted, but performance is less consistent due to dataset imbalance.)

---

## What’s in `artifacts/`

### Model used by the API
The API loads the model from:

```
artifacts/runs/latest/
  config.json
  model_meta.json
  model.joblib
```

- `model.joblib` contains the trained classifier + label encoder.
- `model_meta.json` contains inference metadata (e.g., chunking/config used at prediction time).
- `config.json` stores the run configuration for reproducibility.

### Evaluation outputs
Evaluation files are stored in:

```
artifacts/runs/reports/
  valid_report.txt
  test_report.txt
  confusion_matrix.png
  test_confusion_matrix.csv
  metrics.json
  ...
```

---

## Run with Docker (recommended)

### 1) Build + start the service
From the repo root:

```bash
docker compose up --build
```

This exposes the API on:

- http://localhost:8000

FastAPI interactive documentation:

- http://localhost:8000/docs

### 2) Configuration (environment variables)

`docker-compose.yml` sets:

- `MODEL_DIR=/app/artifacts/runs/latest` (which model artifacts to load)
- `CHUNK_SELECT=first` (how chunks are selected during inference)

The Dockerfile also sets defaults for:

- `HOST=0.0.0.0`
- `PORT=8000`

To swap to a different exported run, change `MODEL_DIR` in `docker-compose.yml` to point to that folder.

---

## Run locally (without Docker)

If you prefer not to use Docker:

```bash
python dialect_package/install_deps.py
python dialect_package/run_app.py
```

This uses the same dependency list as the container (`dialect_package/requirements.txt`).

---

## Reproducibility notes

This project is designed to be reproducible by packaging:
- system-level audio dependencies (`ffmpeg`, `libsndfile1`) in the Docker image,
- Python dependencies via `requirements.txt`,
- exported trained artifacts under `artifacts/runs/latest/`.

---

## Repository structure (high level)

```
.
├── artifacts/
│   └── runs/
│       ├── latest/         # model artifacts used by the API (default)
│       └── reports/        # evaluation outputs
├── dialect_package/
│   ├── install_allosaurus.py
│   ├── install_deps.py
│   ├── requirements.txt
│   ├── run_app.py
│   └── src/
│       ├── dialect_predict.py
│       └── dialect_train.py
├── Dockerfile
├── docker-compose.yml
└── Working Code.ipynb
```

---

## Limitations and future work (data-focused)

Performance varies across dialects primarily due to **class imbalance** and limited examples for some regions. The most impactful improvement would be to collect more recordings for underrepresented dialects and create a more balanced dataset, which would improve fairness and reduce systematic confusion across similar dialects.

---

## Acknowledgements

- RegSpeech12 dataset authors and contributors (Kaggle).
- Hugging Face / PyTorch ecosystem used for speech embeddings and model tooling.
