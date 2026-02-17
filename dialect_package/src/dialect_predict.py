#!/usr/bin/env python3
"""
dialect_predict.py

Inference + API frontend for your saved run folder.

Features:
1) Dialect prediction returns TOP-K predictions + confidence.
2) FastAPI server + simple HTML frontend (no extra files required).
3) Phonetic SOUND timestamp search endpoint (Allosaurus):
   - Given an audio file and a target phone string (ideally space-separated IPA phones),
     returns timestamps for matches found directly in the phone stream.
   - This can match inside other words (substring-by-sound).

Important:
- Dialect prediction behavior matches your notebook by default.
- If (and only if) your run's model_meta.json contains trim_silence/trim_db, inference
  will apply the same trimming to reduce train↔infer mismatch.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torchaudio
import soundfile as sf
import joblib

from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

# ----------------------------
# Optional dependencies for phonetic sound timestamps:
# - allosaurus: universal phone recognizer that can output timestamped phones
# - rapidfuzz: fuzzy matching for phone sequence similarity
# ----------------------------

ALLO_IMPORT_ERROR: Optional[str] = None
try:
    from allosaurus.app import read_recognizer  # type: ignore
except Exception as e:
    read_recognizer = None  # type: ignore
    ALLO_IMPORT_ERROR = repr(e)

try:
    from rapidfuzz import fuzz  # type: ignore
except Exception:
    fuzz = None  # type: ignore

# ----------------------------
# FastAPI (optional if you only use CLI)
# ----------------------------
try:
    from fastapi import FastAPI, File, Form, UploadFile
    from fastapi.responses import HTMLResponse
except Exception:
    FastAPI = None  # type: ignore
    File = None  # type: ignore
    Form = None  # type: ignore
    UploadFile = None  # type: ignore
    HTMLResponse = None  # type: ignore


# ----------------------------
# Audio utils (same as notebook)
# ----------------------------

def load_audio_mono(path: str, target_sr: int) -> torch.Tensor:
    wav, sr = sf.read(path, always_2d=True)
    wav = wav.mean(axis=1)
    wav_t = torch.from_numpy(wav).float()
    if sr != target_sr:
        wav_t = torchaudio.functional.resample(wav_t, sr, target_sr)
    return wav_t


def trim_silence_energy(wav: torch.Tensor, sr: int, trim_db: float = 35.0) -> torch.Tensor:
    # frame RMS energy
    frame = int(0.02 * sr)
    hop = int(0.01 * sr)
    if wav.numel() < frame:
        return wav
    x = wav.unfold(0, frame, hop)
    rms = (x.pow(2).mean(dim=1) + 1e-12).sqrt()
    rms_db = 20 * torch.log10(rms)
    thr = rms_db.max() - trim_db
    keep = rms_db >= thr
    if not keep.any():
        return wav
    idx = keep.nonzero(as_tuple=False).squeeze(1)
    start = int(idx[0].item() * hop)
    end = int(idx[-1].item() * hop + frame)
    end = min(end, wav.numel())
    return wav[start:end]


def make_chunks(wav: torch.Tensor, sr: int, chunk_sec: float, hop_sec: float) -> List[torch.Tensor]:
    L = int(chunk_sec * sr)
    H = int(hop_sec * sr)
    if wav.numel() <= L:
        return [wav]
    chunks: List[torch.Tensor] = []
    for s in range(0, wav.numel() - L + 1, H):
        chunks.append(wav[s : s + L])
    return chunks


def select_chunks_uniform(chunks: List[torch.Tensor], max_chunks: int) -> List[torch.Tensor]:
    if len(chunks) <= max_chunks:
        return chunks
    idx = np.linspace(0, len(chunks) - 1, max_chunks).round().astype(int)
    return [chunks[i] for i in idx]


# ----------------------------
# Embedding pooling (same backbone)
# ----------------------------

def pool_stats3(hidden: torch.Tensor) -> torch.Tensor:
    # hidden: [T, D]
    mean = hidden.mean(dim=0)
    std = hidden.std(dim=0)
    mx = hidden.max(dim=0).values
    return torch.cat([mean, std, mx], dim=0)


# ----------------------------
# Dialect predictor
# ----------------------------

class EmbedPredictor:
    def __init__(self, run_dir: Path, chunk_select: str = "first"):
        self.run_dir = Path(run_dir)
        self.meta = json.loads((self.run_dir / "model_meta.json").read_text())

        pack = joblib.load(self.run_dir / "model.joblib")
        self.clf = pack["model"]
        self.le = pack["label_encoder"]
        self.labels = list(self.le.classes_)

        self.target_sr = int(self.meta["target_sr"])
        self.chunk_sec = float(self.meta["chunk_sec"])
        self.hop_sec = float(self.meta["hop_sec"])
        self.max_chunks = int(self.meta.get("max_chunks_eval", 10))
        self.ssl_name = str(self.meta["ssl_model_name"])
        self.pooling = str(self.meta["pooling"])

        # Train↔infer match (only applies if present in meta; old runs behave like your notebook)
        self.trim_silence = bool(self.meta.get("trim_silence", False))
        self.trim_db = float(self.meta.get("trim_db", 35.0))

        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

        local_only = os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(self.ssl_name, local_files_only=local_only)
        self.ssl = Wav2Vec2Model.from_pretrained(self.ssl_name, local_files_only=local_only).to(self.device).eval()

        self.chunk_select = chunk_select
        if self.chunk_select not in ("first", "uniform"):
            raise ValueError("chunk_select must be 'first' or 'uniform'")

    @torch.no_grad()
    def predict_topk(self, wav_path: str, k: int = 3) -> Dict[str, Any]:
        wav = load_audio_mono(wav_path, self.target_sr)
        if self.trim_silence:
            wav = trim_silence_energy(wav, self.target_sr, self.trim_db)

        chunks = make_chunks(wav, self.target_sr, self.chunk_sec, self.hop_sec)
        if self.chunk_select == "uniform":
            chunks = select_chunks_uniform(chunks, self.max_chunks)
        else:
            chunks = chunks[: self.max_chunks]

        embs = []
        for ch in chunks:
            inputs = self.fe(ch.numpy(), sampling_rate=self.target_sr, return_tensors="pt")
            x = inputs["input_values"].to(self.device)
            out = self.ssl(x).last_hidden_state[0]  # [T,D]
            if self.pooling == "mean":
                emb = out.mean(dim=0)
            else:
                emb = pool_stats3(out)
            embs.append(emb.detach().cpu().numpy())

        X = np.stack(embs, axis=0)

        scores = self.clf.decision_function(X)  # [N, C]
        scores = scores - scores.max(axis=1, keepdims=True)
        exp = np.exp(scores)
        probs = exp / exp.sum(axis=1, keepdims=True)

        p = probs.mean(axis=0)
        top_idx = np.argsort(-p)[: int(k)]
        topk = [{"dialect": self.labels[i], "confidence": float(p[i])} for i in top_idx]
        best = topk[0]
        return {"best": best, "topk": topk, "num_chunks_used": int(len(chunks))}


# ----------------------------
# Phonetic sound timestamp search (Allosaurus)
# ----------------------------

_ALLO_RECOGNIZER = None

def _get_allosaurus_recognizer():
    global _ALLO_RECOGNIZER
    if _ALLO_RECOGNIZER is not None:
        return _ALLO_RECOGNIZER

    if read_recognizer is None:
        msg = (
            "allosaurus import failed. "
            "If you installed it already, this is usually a Python-version or dependency issue. "
            f"Import error was: {ALLO_IMPORT_ERROR}"
        )
        raise RuntimeError(msg)

    _ALLO_RECOGNIZER = read_recognizer()
    return _ALLO_RECOGNIZER


def _normalize_phone_token(p: str) -> str:
    p = (p or "").strip()
    # Remove common IPA diacritics that often cause needless mismatches
    p = p.replace("ˈ", "").replace("ˌ", "").replace("ː", "")
    return p


def _tokenize_query_phones(q: str) -> List[str]:
    q = (q or "").strip()
    q = q.strip("/[](){}<>")
    q = q.replace(",", " ").replace(";", " ")
    q = " ".join(q.split())
    if not q:
        return []
    if " " in q:
        toks = q.split()
    else:
        toks = list(q)  # fallback, less reliable for IPA digraphs
    out: List[str] = []
    for t in toks:
        t = _normalize_phone_token(t)
        if t:
            out.append(t)
    return out


def _parse_allosaurus_timestamp_output(raw: Any) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if raw is None:
        return items

    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return items
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        for ln in lines:
            parts = ln.split()
            if len(parts) < 3:
                continue
            try:
                start = float(parts[0])
                dur = float(parts[1])
                phone = parts[2]
                items.append({"phone": phone, "start": start, "end": start + dur})
            except Exception:
                continue
        if items:
            return items

        toks = s.split()
        for i in range(0, len(toks) - 2, 3):
            try:
                start = float(toks[i])
                dur = float(toks[i + 1])
                phone = toks[i + 2]
                items.append({"phone": phone, "start": start, "end": start + dur})
            except Exception:
                continue
        return items

    if isinstance(raw, (list, tuple)):
        for row in raw:
            try:
                if isinstance(row, (list, tuple)) and len(row) >= 3:
                    start = float(row[0])
                    dur = float(row[1])
                    phone = str(row[2])
                    items.append({"phone": phone, "start": start, "end": start + dur})
            except Exception:
                continue
        return items

    return items


def _dedupe_overlaps(matches: List[Dict[str, Any]], iou_thresh: float = 0.6) -> List[Dict[str, Any]]:
    """Keep highest-score match, remove other matches that overlap strongly in time."""
    if not matches:
        return []

    def iou(a, b) -> float:
        s1, e1 = float(a["start"]), float(a["end"])
        s2, e2 = float(b["start"]), float(b["end"])
        inter = max(0.0, min(e1, e2) - max(s1, s2))
        union = max(e1, e2) - min(s1, s2)
        return 0.0 if union <= 0 else inter / union

    kept: List[Dict[str, Any]] = []
    for m in sorted(matches, key=lambda d: (-float(d["score"]), float(d["start"]))):
        if all(iou(m, k) < iou_thresh for k in kept):
            kept.append(m)

    kept.sort(key=lambda d: float(d["start"]))
    return kept


def find_phonetic_timestamps(
    audio_path: str,
    target_phonetic: str,
    threshold: float = 80.0,
    lang_id: str = "ipa",
    min_phones: int = 5,
    min_duration: float = 0.15,
    max_matches: int = 100,
) -> List[Dict[str, Any]]:
    rec = _get_allosaurus_recognizer()

    query = _tokenize_query_phones(target_phonetic)
    if not query:
        return []
    if len(query) < int(min_phones):
        return []

    wav = load_audio_mono(audio_path, 16000)

    with tempfile.NamedTemporaryFile(prefix="allo_", suffix=".wav", delete=False) as tf:
        tmp_wav = tf.name

    try:
        sf.write(tmp_wav, wav.numpy(), 16000)

        try:
            raw = rec.recognize(tmp_wav, lang_id=str(lang_id), timestamp=True)
        except TypeError:
            raw = rec.recognize(tmp_wav, str(lang_id))

        phones = _parse_allosaurus_timestamp_output(raw)
        if not phones:
            return []

        stream = [_normalize_phone_token(p["phone"]) for p in phones]
        L = len(query)
        out: List[Dict[str, Any]] = []

        for wlen in range(max(1, L - 4), L + 5):  # L-4 ... L+4
            for i in range(0, len(stream) - wlen + 1):
                win = stream[i : i + wlen]

                if fuzz is None:
                    score = 100.0 if win == query else 0.0
                else:
                    score = float(fuzz.ratio(" ".join(win), " ".join(query)))

                if score >= float(threshold):
                    start = float(phones[i]["start"])
                    end = float(phones[i + wlen - 1]["end"])
                    if (end - start) >= float(min_duration):
                        out.append({"start": start, "end": end, "score": score, "phones": win})
                        if len(out) >= int(max_matches):
                            break

            if len(out) >= int(max_matches):
                break

        out.sort(key=lambda d: (float(d["start"]), -float(d["score"])))
        out = _dedupe_overlaps(out, iou_thresh=0.6)
        return out

    finally:
        try:
            os.unlink(tmp_wav)
        except Exception:
            pass


# ----------------------------
# FastAPI app + frontend
# ----------------------------

HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Dialect Model</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; max-width: 900px; margin: 24px auto; padding: 0 16px; }
    .card { border: 1px solid #ddd; border-radius: 12px; padding: 16px; margin: 16px 0; }
    .row { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }
    input[type="file"] { width: 100%; }
    button { padding: 10px 14px; border-radius: 10px; border: 1px solid #ccc; background: #f7f7f7; cursor: pointer; }
    pre { background: #0b1020; color: #d5e2ff; padding: 12px; border-radius: 10px; overflow:auto; }
    .hint { color: #666; font-size: 13px; }
    .title { font-weight: 700; font-size: 18px; margin: 0 0 8px; }
  </style>
</head>
<body>
  <h1>Dialect Model</h1>

  <div class="card">
    <p class="title">1) Dialect prediction (Top-K)</p>
    <div class="row">
      <input id="audio1" type="file" accept=".wav,.flac,.mp3,.m4a" />
    </div>
    <div class="row">
      <label>TopK:</label>
      <input id="topk" type="number" value="3" min="1" max="10" style="width:90px; padding:10px; border-radius:10px; border:1px solid #ccc;" />
      <button onclick="predictDialect()">Predict</button>
    </div>
    <pre id="out1"></pre>
  </div>

  <div class="card">
    <p class="title">2) Keyword timestamp search (sound/phones)</p>
    <p class="hint">
      Best input format is space-separated phones (IPA), e.g. <b>d o ʊ</b> or <b>k æ t</b>.
      This searches the phone stream directly, so it can match inside other words.
    </p>
    <div class="row">
      <input id="audio2" type="file" accept=".wav,.flac,.mp3,.m4a" />
    </div>
    <div class="row">
      <label>Target phones:</label>
      <input id="phon" type="text" style="min-width:280px; flex:1; padding:10px; border-radius:10px; border:1px solid #ccc;" placeholder="e.g., d o ʊ   (space-separated phones recommended)" />
    </div>
    <div class="row">
      <label>Threshold:</label>
      <input id="thr" type="number" value="80" min="0" max="100" style="width:90px; padding:10px; border-radius:10px; border:1px solid #ccc;" />

      <label>Lang:</label>
      <input id="lang" type="text" value="ipa" style="width:90px; padding:10px; border-radius:10px; border:1px solid #ccc;" />

      <label>Min phones:</label>
      <input id="minp" type="number" value="3" min="1" max="20" style="width:90px; padding:10px; border-radius:10px; border:1px solid #ccc;" />

      <label>Min duration (s):</label>
      <input id="mind" type="number" value="0.15" step="0.05" min="0" max="10" style="width:110px; padding:10px; border-radius:10px; border:1px solid #ccc;" />

      <button onclick="findTimestamps()">Find timestamps</button>
    </div>
    <pre id="out2"></pre>
  </div>

<script>
async function predictDialect() {
  const f = document.getElementById("audio1").files[0];
  const k = document.getElementById("topk").value;
  if (!f) return alert("Upload an audio file first.");
  const fd = new FormData();
  fd.append("audio", f);
  fd.append("topk", k);
  const res = await fetch("/predict_dialect", { method: "POST", body: fd });
  const txt = await res.text();
  document.getElementById("out1").textContent = txt;
}

async function findTimestamps() {
  const f = document.getElementById("audio2").files[0];
  const phon = document.getElementById("phon").value;
  const thr = document.getElementById("thr").value;
  const lang = document.getElementById("lang").value;
  const minp = document.getElementById("minp").value;
  const mind = document.getElementById("mind").value;

  if (!f) return alert("Upload an audio file first.");
  if (!phon) return alert("Enter a phone string.");

  const fd = new FormData();
  fd.append("audio", f);
  fd.append("phonetic", phon);
  fd.append("threshold", thr);
  fd.append("lang_id", lang);
  fd.append("min_phones", minp);
  fd.append("min_duration", mind);

  const res = await fetch("/keyword_timestamp", { method: "POST", body: fd });
  const txt = await res.text();
  document.getElementById("out2").textContent = txt;
}
</script>
</body>
</html>
"""


def build_app() -> Any:
    if FastAPI is None:
        raise RuntimeError("FastAPI is not installed. Install fastapi + uvicorn + python-multipart to run the server.")

    model_dir = os.environ.get("MODEL_DIR", "").strip()
    if not model_dir:
        model_dir = "artifacts/runs/latest"

    chunk_select = os.environ.get("CHUNK_SELECT", "first")
    predictor = EmbedPredictor(Path(model_dir), chunk_select=chunk_select)

    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def home():
        return HTML

    @app.post("/predict_dialect")
    async def predict_dialect(audio: UploadFile = File(...), topk: int = Form(3)):
        import tempfile as _tempfile
        filename = getattr(audio, "filename", "") or ""
        suffix = Path(filename).suffix if filename else ".wav"

        with _tempfile.NamedTemporaryFile(prefix="audio_", suffix=suffix or ".wav", delete=False) as tf:
            tmp = Path(tf.name)

        try:
            data = await audio.read()
            tmp.write_bytes(data)
            out = predictor.predict_topk(str(tmp), k=int(topk))
            return out
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

    @app.post("/keyword_timestamp")
    async def keyword_timestamp(
        audio: UploadFile = File(...),
        phonetic: str = Form(...),
        threshold: float = Form(80.0),
        lang_id: str = Form("ipa"),
        min_phones: int = Form(3),
        min_duration: float = Form(0.15),
    ):
        import tempfile as _tempfile
        filename = getattr(audio, "filename", "") or ""
        suffix = Path(filename).suffix if filename else ".wav"

        with _tempfile.NamedTemporaryFile(prefix="audio_", suffix=suffix or ".wav", delete=False) as tf:
            tmp_path = Path(tf.name)

        try:
            data = await audio.read()
            tmp_path.write_bytes(data)

            matches = find_phonetic_timestamps(
                str(tmp_path),
                target_phonetic=phonetic,
                threshold=float(threshold),
                lang_id=str(lang_id),
                min_phones=int(min_phones),
                min_duration=float(min_duration),
            )
            return {"matches": matches, "count": len(matches)}

        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    return app


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=str, required=True, help="Path to artifacts/runs/<run_name> folder.")
    p.add_argument("--wav", type=str, default="", help="Audio path for CLI prediction.")
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--chunk-select", type=str, default="first", choices=["first", "uniform"])
    p.add_argument("--serve", action="store_true", help="Run FastAPI server (requires fastapi+uvicorn).")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.serve:
        os.environ["MODEL_DIR"] = args.run_dir
        os.environ["CHUNK_SELECT"] = args.chunk_select
        app = build_app()

        import uvicorn  # type: ignore
        uvicorn.run(app, host=args.host, port=args.port)
        return

    if not args.wav:
        raise SystemExit("Provide --wav for CLI prediction, or use --serve for API server.")

    pred = EmbedPredictor(Path(args.run_dir), chunk_select=args.chunk_select)
    out = pred.predict_topk(args.wav, k=int(args.topk))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
