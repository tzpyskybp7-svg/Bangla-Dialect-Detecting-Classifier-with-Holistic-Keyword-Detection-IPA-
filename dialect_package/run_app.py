from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

# Respect docker-compose env vars; only set defaults if not provided
os.environ.setdefault("WHISPER_DEVICE", "auto")
os.environ.setdefault("MODEL_DIR", str(ROOT / "model"))
os.environ.setdefault("CHUNK_SELECT", "first")

from dialect_predict import build_app  # noqa: E402

def main():
    app = build_app()

    # IMPORTANT:
    # - In Docker you MUST bind 0.0.0.0
    # - Locally you can still override with HOST=127.0.0.1 if you want
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))

    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    main()
