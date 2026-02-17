import sys
import subprocess

def pip(args):
    print("RUN:", [sys.executable, "-m", "pip"] + args)
    subprocess.check_call([sys.executable, "-m", "pip"] + args)

print("PYTHON:", sys.executable)

# Force install (pin to stable release on PyPI)
pip(["install", "-U", "allosaurus==1.0.2"])

# Verify
import importlib.util
spec = importlib.util.find_spec("allosaurus")
print("allosaurus spec:", spec)

import allosaurus
from allosaurus.app import read_recognizer
print("✅ allosaurus installed + import works")
