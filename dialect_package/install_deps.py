from pathlib import Path
import subprocess
import sys

def pip(args):
    subprocess.check_call([sys.executable, "-m", "pip"] + args)

def main():
    root = Path(__file__).resolve().parent
    req = root / "requirements.txt"

    print("PYTHON:", sys.executable)
    print("REQ FILE:", req)

    pip(["install", "--upgrade", "pip"])
    pip(["install", "-r", str(req), "-v"])

    print("\n✅ Installed dialect_package requirements into THIS interpreter.")
    print("Now run run_app.py with the same interpreter.")

if __name__ == "__main__":
    main()
