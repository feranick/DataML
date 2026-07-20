#!/usr/bin/env bash
#===============================================================================
# setup_dae_spark.sh
#
# Prepares an NVIDIA DGX Spark (GB10, aarch64, CUDA 13) to run DataML_DAE.py
# (Keras 3 on the PyTorch backend) inside an NGC PyTorch container.
#
#   1. Pulls the NGC PyTorch base image  (skipped if already present)
#   2. Builds a derived image with the script's Python dependencies baked in
#      (skipped if already built; force with --rebuild)
#   3. Bakes KERAS_BACKEND=torch into the image and exports it in the launcher
#   4. Runs a verification container that checks the GPU + all imports
#
# After a successful run it prints (and optionally starts) the launch command.
#
# Usage:
#   ./setup_dae_spark.sh [options]
#
# Options:
#   -d, --workdir DIR   Host dir holding DataML_DAE.py, libDataML.py + data
#                       (default: current directory)
#   -t, --tag TAG       NGC PyTorch base image tag (default: 25.11-py3)
#       --rebuild       Rebuild the derived image even if it exists
#       --shell         Drop into an interactive container shell at the end
#   -h, --help          Show this help
#===============================================================================

set -euo pipefail

#------------------------------------------------------------------------------
# Configuration (override via flags)
#------------------------------------------------------------------------------
BASE_TAG="25.11-py3"                 # NGC PyTorch tag verified for GB10 / CUDA 13
BASE_IMAGE=""                        # set after arg parsing
DERIVED_IMAGE="dataml-dae:spark"     # local image with deps baked in
WORKDIR="$(pwd)"
REBUILD=0
DROP_SHELL=0

# Python packages the script needs on top of the base container.
# (torch / numpy / scipy already ship in the NGC PyTorch image — do NOT reinstall torch.)
PY_DEPS="keras scikit-learn pandas h5py matplotlib packaging"

#------------------------------------------------------------------------------
# Pretty output
#------------------------------------------------------------------------------
c_grn=$'\033[1;32m'; c_yel=$'\033[1;33m'; c_red=$'\033[1;31m'; c_rst=$'\033[0m'
info() { printf '%s==>%s %s\n' "$c_grn" "$c_rst" "$*"; }
warn() { printf '%s[!]%s %s\n' "$c_yel" "$c_rst" "$*"; }
err()  { printf '%s[x]%s %s\n' "$c_red" "$c_rst" "$*" >&2; }
die()  { err "$*"; exit 1; }

#------------------------------------------------------------------------------
# Argument parsing
#------------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--workdir) WORKDIR="$2"; shift 2 ;;
    -t|--tag)     BASE_TAG="$2"; shift 2 ;;
    --rebuild)    REBUILD=1; shift ;;
    --shell)      DROP_SHELL=1; shift ;;
    -h|--help)    sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)            die "Unknown option: $1 (use --help)" ;;
  esac
done

BASE_IMAGE="nvcr.io/nvidia/pytorch:${BASE_TAG}"
WORKDIR="$(cd "$WORKDIR" && pwd)"    # normalize to absolute path

#------------------------------------------------------------------------------
# Docker availability + permissions (handles the "permission denied" case)
#------------------------------------------------------------------------------
command -v docker >/dev/null 2>&1 || die "docker is not installed or not on PATH."

DOCKER="docker"
if ! docker info >/dev/null 2>&1; then
  if sudo -n docker info >/dev/null 2>&1 || sudo docker info >/dev/null 2>&1; then
    warn "Cannot reach the Docker daemon as \$USER; falling back to 'sudo docker'."
    warn "To avoid sudo permanently:  sudo usermod -aG docker \$USER  (then re-login)"
    DOCKER="sudo docker"
  else
    die "Cannot talk to the Docker daemon (even with sudo). Is dockerd running?  sudo systemctl status docker"
  fi
fi

# Sanity: warn if the NVIDIA container runtime isn't wired up (GPU verify would fail).
if ! $DOCKER info 2>/dev/null | grep -qi 'nvidia'; then
  warn "NVIDIA container runtime not detected in 'docker info'."
  warn "If GPU verification fails, install/enable the NVIDIA Container Toolkit."
fi

info "Base image     : $BASE_IMAGE"
info "Derived image  : $DERIVED_IMAGE"
info "Work directory : $WORKDIR"

#------------------------------------------------------------------------------
# Step 1 — Pull base image (skip if already present)
#------------------------------------------------------------------------------
if $DOCKER image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
  info "[1/4] Base image already present — skipping pull."
else
  info "[1/4] Pulling base image (this can take a while)…"
  if ! $DOCKER pull "$BASE_IMAGE"; then
    err "Pull failed. If this is an auth error, log in first:  $DOCKER login nvcr.io"
    die "  (username: \$oauthtoken, password: your NGC API key)"
  fi
fi

#------------------------------------------------------------------------------
# Step 2 & 3 — Build derived image with deps + KERAS_BACKEND baked in
#------------------------------------------------------------------------------
if [[ "$REBUILD" -eq 0 ]] && $DOCKER image inspect "$DERIVED_IMAGE" >/dev/null 2>&1; then
  info "[2/4] Derived image already built — skipping (use --rebuild to force)."
else
  info "[2/4] Building derived image with deps: $PY_DEPS"
  # Build context is empty (piped via stdin); only FROM + ENV + RUN are used.
  $DOCKER build -t "$DERIVED_IMAGE" - <<DOCKERFILE
FROM ${BASE_IMAGE}
# Step 3: make the Keras backend selection permanent inside the image.
ENV KERAS_BACKEND=torch
# Step 2: install the script's Python dependencies (torch is left untouched).
RUN pip install --no-cache-dir ${PY_DEPS}
WORKDIR /workspace/dae
DOCKERFILE
  info "[2/4] Derived image built. KERAS_BACKEND=torch baked in."
fi

#------------------------------------------------------------------------------
# Step 4 — Verify everything inside a container
#------------------------------------------------------------------------------
info "[4/4] Verifying GPU + Python environment inside the container…"

set +e
$DOCKER run --gpus all --rm -i "$DERIVED_IMAGE" python - <<'PYEOF'
import os, sys

ok = True
def check(label, fn):
    global ok
    try:
        val = fn()
        print(f"  [✓] {label}: {val}")
    except Exception as e:
        ok = False
        print(f"  [x] {label}: FAILED -> {e}")

print("\n  --- Backend / framework ---")
check("KERAS_BACKEND env", lambda: os.environ.get("KERAS_BACKEND", "<unset>"))

import torch
check("torch version", lambda: torch.__version__)
check("CUDA available", lambda: torch.cuda.is_available())
if torch.cuda.is_available():
    check("GPU name", lambda: torch.cuda.get_device_name(0))
    check("CUDA (torch) version", lambda: torch.version.cuda)
    # sm_121 vs sm_120: a real compute test proves binary compatibility.
    def _matmul():
        a = torch.randn(512, 512, device="cuda")
        b = torch.randn(512, 512, device="cuda")
        return tuple((a @ b).shape)
    check("GPU matmul (sm_121 kernel test)", _matmul)
else:
    ok = False
    print("  [x] No CUDA device visible to torch — check --gpus all / NVIDIA toolkit.")

print("\n  --- Keras 3 (backend-agnostic) ---")
import keras
check("keras version", lambda: keras.__version__)
check("keras backend", lambda: keras.backend.backend())
if keras.backend.backend() != "torch":
    ok = False
    print("  [x] Keras backend is not 'torch' — KERAS_BACKEND not applied.")
# Prove Keras ops run (they dispatch to torch/CUDA under the hood).
check("keras.ops smoke test", lambda: float(keras.ops.sum(keras.ops.convert_to_tensor([1.0, 2.0, 3.0]))))

print("\n  --- Script dependencies ---")
for mod in ["numpy", "scipy", "sklearn", "pandas", "h5py", "matplotlib", "configparser"]:
    check(f"import {mod}", lambda m=mod: __import__(m).__name__)

print()
if ok:
    print("  RESULT: \033[1;32mALL CHECKS PASSED\033[0m")
    sys.exit(0)
else:
    print("  RESULT: \033[1;31mONE OR MORE CHECKS FAILED\033[0m")
    sys.exit(1)
PYEOF
VERIFY_RC=$?
set -e

echo
if [[ "$VERIFY_RC" -ne 0 ]]; then
  die "Verification failed (exit $VERIFY_RC). See the [x] lines above."
fi
info "Verification passed."
warn "A 'sm_121 > sm_120' capability warning from torch is expected and harmless."

#------------------------------------------------------------------------------
# Done — show how to launch, and optionally drop into a shell
#------------------------------------------------------------------------------
RUN_CMD="$DOCKER run --gpus all -it --rm --ipc=host \\
    -v \"$WORKDIR\":/workspace/dae -w /workspace/dae \\
    $DERIVED_IMAGE"

cat <<EOF

${c_grn}Setup complete.${c_rst}  Everything is baked into the image '${DERIVED_IMAGE}'.

To launch the container (KERAS_BACKEND=torch is already set inside):

    ${RUN_CMD}

Then, at the container prompt, run your script — e.g.:

    python DataML_DAE.py -t <learningFile>     # train
    python DataML_DAE.py -a <learningFile>     # augment
    python DataML_DAE.py -g <csvFile>          # generate

EOF

if [[ "$DROP_SHELL" -eq 1 ]]; then
  info "Dropping into an interactive container shell…"
  eval "$RUN_CMD"
fi
