#!/usr/bin/env python3
"""
dialect_train.py

Train a dialect classifier using SSL embeddings (wav2vec2-xls-r-300m by default)
and a lightweight downstream classifier (LinearSVC or XGBoost).

This script is a direct, CLI-friendly extraction of your notebook logic.
Defaults are chosen to match the notebook as closely as possible.

Outputs (per run) under: artifacts/runs/<timestamp>_<strategy>/
- config.json
- manifest.csv
- cache/emb_*.npz
- model.joblib
- model_meta.json
- valid_report.txt
- metrics.json
- confusion_matrix.png
- misclassified_valid.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import soundfile as sf

import torch
import torchaudio

from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC

import joblib
import matplotlib.pyplot as plt

try:
    import xgboost as xgb  # type: ignore
except Exception:
    xgb = None


# ----------------------------
# Config
# ----------------------------

@dataclass
class TrainConfig:
    project_root: str = "."
    data_root: str = ""
    artifacts_dir: str = "artifacts"
    run_name: str = ""

    filename_regex: str = r"^(train|valid|test)_(.+?)_(\d+)\.wav$"

    target_sr: int = 16000
    chunk_sec: float = 4.0
    hop_sec: float = 2.0
    max_chunks_per_file_train: int = 6
    max_chunks_per_file_eval: int = 10
    trim_silence: bool = True
    trim_db: float = 35.0

    ssl_model_name: str = "facebook/wav2vec2-xls-r-300m"
    pooling: str = "stats3"          # "mean" or "stats3"
    cache_embeddings: bool = True

    strategy: str = "embed_linear_svc"  # "embed_linear_svc" | "embed_xgboost"

    # LinearSVC
    svc_max_iter: int = 20000  # safe: avoids early stop (you saw ConvergenceWarning)

    # XGBoost
    xgb_n_estimators: int = 800
    xgb_max_depth: int = 8
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.9
    xgb_colsample: float = 0.9

    seed: int = 7


# ----------------------------
# Utilities (matching notebook)
# ----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def load_audio_mono(path: str, target_sr: int) -> torch.Tensor:
    wav, sr = sf.read(path, always_2d=True)
    wav = wav.mean(axis=1)
    wav_t = torch.tensor(wav, dtype=torch.float32)
    if sr != target_sr:
        wav_t = torchaudio.functional.resample(wav_t, sr, target_sr)
    return wav_t


def trim_silence_energy(wav: torch.Tensor, sr: int, trim_db: float) -> torch.Tensor:
    # identical to notebook
    if wav.numel() < sr // 2:
        return wav
    frame = int(sr * 0.02)
    hop = int(sr * 0.01)
    x = wav.unsqueeze(0)
    frames = x.unfold(1, frame, hop)
    rms = torch.sqrt(torch.mean(frames ** 2, dim=-1) + 1e-9).squeeze(0)
    rms_db = 20 * torch.log10(rms + 1e-9)
    thr = rms_db.max() - trim_db
    keep = (rms_db >= thr).nonzero(as_tuple=False).squeeze(-1)
    if keep.numel() == 0:
        return wav
    start = int(keep.min().item()) * hop
    end = min(wav.numel(), int(keep.max().item()) * hop + frame)
    return wav[start:end]


def make_chunks(wav: torch.Tensor, sr: int, chunk_sec: float, hop_sec: float) -> List[torch.Tensor]:
    chunk = int(sr * chunk_sec)
    hop = int(sr * hop_sec)
    if wav.numel() <= chunk:
        return [torch.nn.functional.pad(wav, (0, chunk - wav.numel()))]
    out: List[torch.Tensor] = []
    for start in range(0, wav.numel() - chunk + 1, hop):
        out.append(wav[start:start + chunk])
    return out if out else [wav[:chunk]]


def sample_chunks(chunks: List[torch.Tensor], k: int, seed: int) -> List[torch.Tensor]:
    if len(chunks) <= k:
        return chunks
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(chunks), size=k, replace=False)
    return [chunks[i] for i in idx]


def plot_confusion(cm: np.ndarray, labels: List[str], out_png: Path) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    plt.colorbar(im, ax=ax)
    ticks = np.arange(len(labels))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


# ----------------------------
# Manifest
# ----------------------------

def find_split_dir(base: Path, split: str) -> Path:
    for p in base.rglob(split):
        if p.is_dir() and any(p.glob("*.wav")):
            return p
    raise FileNotFoundError(f"Could not find '{split}' folder with wavs under {base}")


def build_manifest(data_root: Path, filename_regex: str) -> pd.DataFrame:
    rgx = re.compile(filename_regex)
    train_dir = find_split_dir(data_root, "train")
    valid_dir = find_split_dir(data_root, "valid")
    test_dir = find_split_dir(data_root, "test")

    def list_wavs(split_name: str, split_dir: Path) -> pd.DataFrame:
        rows = []
        for p in sorted(split_dir.glob("*.wav")):
            fn = p.name
            m = rgx.match(fn)
            dialect = m.group(2) if m else "unknown"
            rows.append({"path": str(p), "file": fn, "split": split_name, "dialect": dialect})
        return pd.DataFrame(rows)

    df = pd.concat(
        [list_wavs("train", train_dir), list_wavs("valid", valid_dir), list_wavs("test", test_dir)],
        ignore_index=True,
    )
    return df


# ----------------------------
# Embeddings + cache
# ----------------------------

class SSLEncoder:
    def __init__(self, model_name: str, device: str, target_sr: int, pooling: str):
        self.device = device
        self.target_sr = target_sr
        self.pooling = pooling
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def encode_chunk(self, wav_chunk: torch.Tensor) -> np.ndarray:
        inputs = self.processor(
            wav_chunk.numpy(),
            sampling_rate=self.target_sr,
            return_tensors="pt",
            padding=True,
        )
        x = inputs["input_values"].to(self.device)
        h = self.model(x).last_hidden_state.squeeze(0)

        if self.pooling == "mean":
            emb = h.mean(dim=0)
            return emb.cpu().numpy().astype(np.float32)

        if self.pooling == "stats3":
            mu = h.mean(dim=0)
            sd = h.std(dim=0)
            mx = h.max(dim=0).values
            emb = torch.cat([mu, sd, mx], dim=0)
            return emb.cpu().numpy().astype(np.float32)

        raise ValueError("pooling must be 'mean' or 'stats3'")


def embed_dataset_chunks(
    cfg: TrainConfig,
    df: pd.DataFrame,
    split: str,
    encoder: SSLEncoder,
    cache_dir: Path,
    seed_offset: int,
):
    df_split = df[df["split"] == split].reset_index(drop=True)

    key = sha1(
        json.dumps(
            {
                "model": cfg.ssl_model_name,
                "pooling": cfg.pooling,
                "split": split,
                "chunk_sec": cfg.chunk_sec,
                "hop_sec": cfg.hop_sec,
                "max_chunks": cfg.max_chunks_per_file_train if split == "train" else cfg.max_chunks_per_file_eval,
                "trim": cfg.trim_silence,
                "trim_db": cfg.trim_db,
                "data_root": str(Path(cfg.data_root).resolve()),
            },
            sort_keys=True,
        )
    )
    cache_path = cache_dir / f"emb_{key}.npz"

    if cfg.cache_embeddings and cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        return z["X"], z["y"], z["file_ids"]

    X_list, y_list, file_list = [], [], []
    max_chunks = cfg.max_chunks_per_file_train if split == "train" else cfg.max_chunks_per_file_eval

    for i, r in tqdm(df_split.iterrows(), total=len(df_split), desc=f"Embedding {split}"):
        wav = load_audio_mono(r["path"], cfg.target_sr)
        if cfg.trim_silence:
            wav = trim_silence_energy(wav, cfg.target_sr, cfg.trim_db)

        chunks = make_chunks(wav, cfg.target_sr, cfg.chunk_sec, cfg.hop_sec)
        chunks = sample_chunks(chunks, max_chunks, seed=cfg.seed + seed_offset + i)

        for c in chunks:
            X_list.append(encoder.encode_chunk(c))
            y_list.append(r["dialect"])
            file_list.append(r["path"])

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list)
    file_ids = np.array(file_list)

    if cfg.cache_embeddings:
        np.savez_compressed(cache_path, X=X, y=y, file_ids=file_ids)

    return X, y, file_ids


def aggregate_probs_by_file(file_ids: np.ndarray, probs: np.ndarray):
    dfp = pd.DataFrame({"file": file_ids})
    for j in range(probs.shape[1]):
        dfp[f"p{j}"] = probs[:, j]
    grp = dfp.groupby("file", sort=False).mean(numeric_only=True).reset_index()
    files = grp["file"].values
    pcols = [c for c in grp.columns if c.startswith("p")]
    return files, grp[pcols].values


# ----------------------------
# Training
# ----------------------------

def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def train_embed_model(cfg: TrainConfig, df: pd.DataFrame, run_dir: Path, encoder: SSLEncoder, cache_dir: Path):
    Xtr, ytr, ftr = embed_dataset_chunks(cfg, df, "train", encoder, cache_dir, seed_offset=100)
    Xva, yva, fva = embed_dataset_chunks(cfg, df, "valid", encoder, cache_dir, seed_offset=200)

    le = LabelEncoder()
    ytr_i = le.fit_transform(ytr)
    labels = list(le.classes_)
    n_classes = len(labels)

    if cfg.strategy == "embed_linear_svc":
        clf = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svc", LinearSVC(class_weight="balanced", max_iter=cfg.svc_max_iter)),
            ]
        )
        clf.fit(Xtr, ytr_i)

        dec = clf.decision_function(Xva)
        if dec.ndim == 1:
            dec = np.stack([-dec, dec], axis=1)
        probs_chunks = _softmax_np(dec)

    elif cfg.strategy == "embed_xgboost":
        if xgb is None:
            raise RuntimeError("xgboost not installed. Install with: pip install xgboost")

        ytr_i = le.transform(ytr)

        counts = np.bincount(ytr_i, minlength=n_classes)
        inv = counts.max() / np.maximum(counts, 1)
        w = inv[ytr_i].astype(np.float32)

        clf = xgb.XGBClassifier(
            n_estimators=cfg.xgb_n_estimators,
            max_depth=cfg.xgb_max_depth,
            learning_rate=cfg.xgb_learning_rate,
            subsample=cfg.xgb_subsample,
            colsample_bytree=cfg.xgb_colsample,
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
            tree_method="hist",
        )
        clf.fit(Xtr, ytr_i, sample_weight=w, eval_set=[(Xva, le.transform(yva))], verbose=False, early_stopping_rounds=50)
        probs_chunks = clf.predict_proba(Xva)

    else:
        raise ValueError("strategy must be embed_linear_svc or embed_xgboost")

    # aggregate chunk probs -> file probs
    files_va, probs_va = aggregate_probs_by_file(fva, probs_chunks)

    valid_map = df[df["split"] == "valid"].set_index("path")["dialect"].to_dict()
    yva_file = np.array([valid_map[f] for f in files_va])
    yva_file_i = le.transform(yva_file)

    pred_i = probs_va.argmax(axis=1)
    acc = accuracy_score(yva_file_i, pred_i)
    f1m = f1_score(yva_file_i, pred_i, average="macro")

    report = classification_report(yva_file_i, pred_i, target_names=labels, digits=4)
    cm = confusion_matrix(yva_file_i, pred_i)

    (run_dir / "valid_report.txt").write_text(report)
    (run_dir / "metrics.json").write_text(
        json.dumps({"valid_accuracy": float(acc), "valid_macro_f1": float(f1m)}, indent=2)
    )
    plot_confusion(cm, labels, run_dir / "confusion_matrix.png")

    miss = pd.DataFrame(
        {
            "file": files_va,
            "true": le.inverse_transform(yva_file_i),
            "pred": le.inverse_transform(pred_i),
            "confidence": probs_va.max(axis=1),
        }
    )
    miss = miss[miss["true"] != miss["pred"]].sort_values("confidence", ascending=False)
    miss.to_csv(run_dir / "misclassified_valid.csv", index=False)

    joblib.dump({"model": clf, "label_encoder": le}, run_dir / "model.joblib")

    # NOTE: include trim settings so inference can match training (but old runs still work without them).
    (run_dir / "model_meta.json").write_text(
        json.dumps(
            {
                "strategy": cfg.strategy,
                "ssl_model_name": cfg.ssl_model_name,
                "pooling": cfg.pooling,
                "target_sr": cfg.target_sr,
                "chunk_sec": cfg.chunk_sec,
                "hop_sec": cfg.hop_sec,
                "max_chunks_eval": cfg.max_chunks_per_file_eval,
                "trim_silence": cfg.trim_silence,
                "trim_db": cfg.trim_db,
            },
            indent=2,
        )
    )

    print(f"✅ Saved run: {run_dir}")
    print("Valid accuracy:", float(acc))
    print("Valid macro-F1 :", float(f1m))
    return float(acc), float(f1m)


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, required=True, help="Dataset root containing train/valid/test folders.")
    p.add_argument("--strategy", type=str, default="embed_linear_svc", choices=["embed_linear_svc", "embed_xgboost"])
    p.add_argument("--ssl-model-name", type=str, default="facebook/wav2vec2-xls-r-300m")
    p.add_argument("--pooling", type=str, default="stats3", choices=["mean", "stats3"])
    p.add_argument("--artifacts-dir", type=str, default="artifacts")
    p.add_argument("--run-name", type=str, default="")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--no-trim", action="store_true", help="Disable silence trimming (not recommended if you trained with it).")
    p.add_argument("--trim-db", type=float, default=35.0)
    p.add_argument("--svc-max-iter", type=int, default=20000)
    p.add_argument("--no-cache", action="store_true", help="Disable embedding cache.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    project_root = Path(".").resolve()
    artifacts_dir = project_root / args.artifacts_dir
    runs_dir = artifacts_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_name = args.run_name.strip() or (time.strftime("%Y%m%d_%H%M%S") + "_" + args.strategy)
    run_dir = runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = TrainConfig(
        project_root=str(project_root),
        data_root=str(Path(args.data_root).expanduser()),
        artifacts_dir=str(artifacts_dir),
        run_name=run_name,
        strategy=args.strategy,
        ssl_model_name=args.ssl_model_name,
        pooling=args.pooling,
        seed=args.seed,
        trim_silence=not args.no_trim,
        trim_db=float(args.trim_db),
        svc_max_iter=int(args.svc_max_iter),
        cache_embeddings=not args.no_cache,
    )

    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    print("Saved config:", run_dir / "config.json")

    set_seed(cfg.seed)
    device = get_device()
    print("DEVICE:", device)

    df = build_manifest(Path(cfg.data_root), cfg.filename_regex)
    df.to_csv(run_dir / "manifest.csv", index=False)
    print("Saved manifest:", run_dir / "manifest.csv")
    print("Total clips:", len(df))
    print("Dialects:", df["dialect"].nunique())

    cache_dir = run_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    encoder = SSLEncoder(cfg.ssl_model_name, device, cfg.target_sr, cfg.pooling)
    print("Encoder ready.")

    train_embed_model(cfg, df, run_dir, encoder, cache_dir)

    print("\nRun folder:", run_dir)


if __name__ == "__main__":
    main()
