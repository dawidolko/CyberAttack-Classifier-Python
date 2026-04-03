#!/usr/bin/env bash
# =============================================================
#  Cyber Security Attacks Classifier — Launcher (Linux / macOS)
# =============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="venv"
PYTHON=""

echo "============================================================"
echo "  🛡️  Cyber Security Attacks Classifier"
echo "  Random Forest — ML Pipeline + Streamlit Dashboard"
echo "============================================================"
echo ""

# --- Find Python 3.10+ ---
find_python() {
    for cmd in python3.14 python3.13 python3.12 python3.11 python3.10 python3; do
        if command -v "$cmd" &>/dev/null; then
            version=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
            major=$(echo "$version" | cut -d. -f1)
            minor=$(echo "$version" | cut -d. -f2)
            if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ] && [ "$minor" -le 14 ]; then
                PYTHON="$cmd"
                echo "   Found Python $version ($cmd)"
                return 0
            fi
        fi
    done
    echo "❌ Error: Python 3.10+ is required but not found."
    echo "   Please install Python 3.12: https://www.python.org/downloads/"
    exit 1
}

echo "🔍 Step 1/4: Checking Python installation..."
find_python

# --- Create / activate virtual environment ---
echo ""
echo "📦 Step 2/4: Setting up virtual environment..."
if [ ! -d "$VENV_DIR" ]; then
    echo "   Creating virtual environment..."
    $PYTHON -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "   Installing dependencies (pip progress below)..."
pip install --upgrade pip 2>/dev/null
pip install -r requirements.txt
echo "   ✅ Dependencies installed"

# --- Run ML Pipeline ---
echo ""
echo "🚀 Step 3/4: Running ML Pipeline..."
echo ""
python pipeline.py

# --- Launch Streamlit ---
echo ""
echo "🌐 Step 4/4: Launching Streamlit Dashboard..."
echo ""
echo "   Dashboard will open in your browser at: http://localhost:8501"
echo "   Press Ctrl+C to stop the server."
echo ""
streamlit run app.py --server.headless true
