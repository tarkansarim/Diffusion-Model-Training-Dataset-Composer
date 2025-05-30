#!/bin/bash

# Diffusion Training Dataset Composer Linux Launcher
# Robust version: always uses venv, installs dependencies, launches app, works from any directory
set -euo pipefail

# Resolve script location and cd to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Debug: Print current working directory
pwd

# Fix: Only check $1 if set, to avoid 'unbound variable' error
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: ./run_linux.sh"
    echo "Runs the PyQt5 dataset composer tool with python3 in a virtual environment (auto-created if missing)."
    exit 0
fi

# Ensure QT_QPA_PLATFORM is set for PyQt5 compatibility
export QT_QPA_PLATFORM=${QT_QPA_PLATFORM:-xcb}

# Check for python3
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 not found. Please install Python 3.8+ and try again."
    exit 1
fi

# Always use venv in project root
if [[ ! -d venv ]]; then
    echo "[INFO] Creating Python virtual environment in ./venv..."
    python3 -m venv venv || { echo "[ERROR] Failed to create venv."; exit 1; }
fi

# Activate venv
source venv/bin/activate

# Print debug info
echo "[DEBUG] Python executable: $(which python)"
echo "[DEBUG] Pip executable: $(which pip)"
python --version
pip --version

# Ensure pip is available and up to date
python -m ensurepip --upgrade || true
python -m pip install --upgrade pip

# Always install requirements (ensures venv is correct)
if [[ -f requirements.txt ]]; then
    python -m pip install -r requirements.txt || { echo "[ERROR] pip install failed"; exit 1; }
else
    python -m pip install PyQt5 Pillow || { echo "[ERROR] pip install failed"; exit 1; }
fi

echo "[DEBUG] Installed packages:"
pip list

# Fix SyntaxWarning in image_sampler_tool.py if present
if grep -q "invalid escape sequence '\/'" image_sampler_tool.py 2>/dev/null; then
    sed -i "s/\'\\/\:\*\?\"<>|\'/\\/\:\*\?\"<>|/g" image_sampler_tool.py
fi

# Run the tool using venv's python
python image_sampler_tool.py 