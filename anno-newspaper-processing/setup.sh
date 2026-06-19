#!/usr/bin/env bash
#
# setup.sh — environment setup for the eynollah notebook project
#
# Creates two conda/mamba environments:
#   1. $EYN_ENV  — runs eynollah (TensorFlow 2.12 + CUDA 11.8). The notebook
#                  calls into this env via subprocess; it is NOT a Jupyter kernel.
#   2. $NB_ENV   — the environment the notebook itself runs in
#                  (Jupyter + OCR / data libraries).
#
# It also:
#   - installs system Tesseract 5 + dev headers (so tesserocr builds against it),
#   - downloads the eynollah models into ./model_dir,
#   - installs the frak2021 Tesseract model into the system tessdata directory.
#
# Tested target: Ubuntu 24.04 (Tesseract 5).
# On Ubuntu 22.04 the default Tesseract is 4.x — see the note at the bottom.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh

set -euo pipefail

# ============================================================================
# Configuration — adjust as needed
# ============================================================================
EYN_ENV="eyn_env"                 # environment for eynollah (keep in sync w/ notebook)
NB_ENV="anno_env"             # environment the notebook runs in
NB_PY_VERSION="3.11"              # Python version for the notebook environment

EYNOLLAH_REPO="https://github.com/Var3n/eynollah.git"
EYNOLLAH_BRANCH="eynollah_v0.7.0"

EYN_MODELS_URL="https://zenodo.org/records/17580627/files/models_all_v0_7_0.zip?download=1"
TESS_MODEL_URL="https://ub-backup.bib.uni-mannheim.de/~stweil/tesstrain/frak2021/tessdata_best/frak2021-0.905.traineddata"
TESS_MODEL_FILE="frak2021-0.905.traineddata"   # Tesseract language code = filename without .traineddata

# Directory this script lives in (models go into ./model_dir next to it)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$SCRIPT_DIR/model_dir"

# Use sudo only if we are not already root (e.g. inside a container)
SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    SUDO="sudo"
fi

# ============================================================================
# Detect mamba / conda
# ============================================================================
if command -v mamba >/dev/null 2>&1; then
    CONDA=mamba
elif command -v conda >/dev/null 2>&1; then
    CONDA=conda
else
    echo "ERROR: neither 'mamba' nor 'conda' was found on PATH." >&2
    echo "       Install Miniforge/Mambaforge first: https://github.com/conda-forge/miniforge" >&2
    exit 1
fi
echo ">> Using '$CONDA' for environment management."

# ============================================================================
# 1. eynollah environment (TensorFlow 2.12 + CUDA 11.8)
# ============================================================================
echo ">> [1/5] Creating '$EYN_ENV' (eynollah / TensorFlow) ..."
"$CONDA" create -y --name "$EYN_ENV" python=3.10
"$CONDA" install -y -n "$EYN_ENV" -c conda-forge cudatoolkit=11.8 cudnn=8.6
"$CONDA" run -n "$EYN_ENV" pip install "tensorflow==2.12.*"

# Custom eynollah fork (specific branch) instead of the PyPI release
"$CONDA" run -n "$EYN_ENV" pip install "git+${EYNOLLAH_REPO}@${EYNOLLAH_BRANCH}"

# Make TensorFlow find the conda-provided CUDA libraries when the env is activated.
# (TF 2.12 needs cudatoolkit/cudnn on LD_LIBRARY_PATH; this also applies when the
#  notebook launches eynollah via `conda run -n eyn_env ...`.)
EYN_PREFIX="$("$CONDA" run -n "$EYN_ENV" bash -c 'echo "$CONDA_PREFIX"')"
mkdir -p "$EYN_PREFIX/etc/conda/activate.d"
cat > "$EYN_PREFIX/etc/conda/activate.d/cuda_ld.sh" <<'EOF'
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
EOF

# ============================================================================
# 2. System Tesseract 5 + build/runtime dependencies
# ============================================================================
echo ">> [2/5] Installing system Tesseract and build dependencies ..."
$SUDO apt-get update
$SUDO apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev \
    pkg-config \
    build-essential \
    wget \
    unzip \
    libgl1 \
    libglib2.0-0          # the last two keep eynollah's OpenCV happy at runtime

# Detect the active tessdata directory; fall back to the standard Tesseract 5 path.
FOUND="$(find /usr/share/tesseract-ocr -name '*.traineddata' 2>/dev/null | head -n1 || true)"
if [ -n "$FOUND" ]; then
    TESSDATA_DIR="$(dirname "$FOUND")"
else
    TESSDATA_DIR="/usr/share/tesseract-ocr/5/tessdata"
fi
echo ">> tessdata directory: $TESSDATA_DIR"

# ============================================================================
# 3. Notebook environment (Jupyter + everything else)
# ============================================================================
echo ">> [3/5] Creating '$NB_ENV' (notebook / OCR libraries) ..."
"$CONDA" create -y --name "$NB_ENV" python="$NB_PY_VERSION"

# tesserocr is built from source against the system libtesseract installed above.
"$CONDA" run -n "$NB_ENV" pip install \
    jupyterlab \
    ipykernel \
    tqdm \
    pandas \
    numpy \
    Pillow \
    transformers \
    ocrd_models \
    ocrd_utils \
    tesserocr

# Register a Jupyter kernel so the notebook can select this environment.
"$CONDA" run -n "$NB_ENV" python -m ipykernel install --user \
    --name "$NB_ENV" --display-name "Python ($NB_ENV)"

# Point this env's Tesseract at the system tessdata directory (where frak2021 lands).
NB_PREFIX="$("$CONDA" run -n "$NB_ENV" bash -c 'echo "$CONDA_PREFIX"')"
mkdir -p "$NB_PREFIX/etc/conda/activate.d"
cat > "$NB_PREFIX/etc/conda/activate.d/tessdata.sh" <<EOF
export TESSDATA_PREFIX="$TESSDATA_DIR"
EOF

# ============================================================================
# 4. Download eynollah models into ./model_dir
# ============================================================================
echo ">> [4/5] Downloading eynollah models into $MODEL_DIR ..."
mkdir -p "$MODEL_DIR"
TMP_ZIP="$SCRIPT_DIR/models_all_v0_7_0.zip"
wget -O "$TMP_ZIP" "$EYN_MODELS_URL"
unzip -o "$TMP_ZIP" -d "$MODEL_DIR"
rm -f "$TMP_ZIP"
# NOTE: if the archive contains a single top-level folder, the models will end up
# in $MODEL_DIR/<that-folder>/. Run `ls "$MODEL_DIR"` to confirm the layout matches
# what the notebook expects, and adjust the model path in the notebook if needed.

# ============================================================================
# 5. Install the frak2021 Tesseract model into the tessdata directory
# ============================================================================
echo ">> [5/5] Installing frak2021 Tesseract model ..."
$SUDO wget -O "$TESSDATA_DIR/$TESS_MODEL_FILE" "$TESS_MODEL_URL"
# Tip: to use a shorter language name, rename to frak2021.traineddata, e.g.:
#   $SUDO mv "$TESSDATA_DIR/$TESS_MODEL_FILE" "$TESSDATA_DIR/frak2021.traineddata"

# ============================================================================
# Done
# ============================================================================
echo
echo "Setup complete."
echo "  eynollah env     : $EYN_ENV  (used via subprocess)"
echo "  notebook env     : $NB_ENV   (Jupyter kernel: 'Python ($NB_ENV)')"
echo "  eynollah models  : $MODEL_DIR"
echo "  tesseract model  : $TESSDATA_DIR/$TESS_MODEL_FILE"
echo "  tesseract lang   : ${TESS_MODEL_FILE%.traineddata}"
echo
echo "Start the notebook with:  conda run -n $NB_ENV jupyter lab"
echo "and select the 'Python ($NB_ENV)' kernel."

# ----------------------------------------------------------------------------
# Ubuntu 22.04 note:
#   The default Tesseract there is 4.x (tessdata at /usr/share/tesseract-ocr/4.00).
#   To get Tesseract 5 + the /5/ path, add the PPA before step 2:
#       sudo add-apt-repository -y ppa:alex-p/tesseract-ocr5
#       sudo apt-get update
# ----------------------------------------------------------------------------
