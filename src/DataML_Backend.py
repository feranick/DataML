# -*- coding: utf-8 -*-
'''
**************************************************
* DataML_Backend - shared device / backend config
* version: 2026.07.20.1
* Backend-agnostic (TensorFlow / PyTorch / JAX), SLURM-aware.
* By: Nicola Ferralis <feranick@hotmail.com>
**************************************************

Shared, backend-agnostic device configuration for the DataML scripts
(DataML_DAE, DataML_VAE, ...). Import this module BEFORE `import keras`:

    from DataML_Backend import configureDevices   # selects device at import
    import keras
    ...
    def main():
        configureDevices()    # applies backend-specific memory tuning
        ...

WHY IMPORT-BEFORE-KERAS MATTERS
    Device SELECTION is done by setting CUDA_VISIBLE_DEVICES (and JAX env vars),
    which must be in place BEFORE the Keras backend initializes CUDA. Importing
    this module performs that selection as a side effect, so simply importing it
    first guarantees the correct ordering - even in scripts that `import keras`
    at module top level.

RUNTIME CONTROLS (environment variables)
    KERAS_BACKEND      : tensorflow (default) | torch | jax
    DEVICE_PREFERENCE  : auto (default) | cpu | gpu | <int GPU index>

SELECTION PRIORITY (highest first)
    1. DEVICE_PREFERENCE=cpu        -> force CPU.
    2. DEVICE_PREFERENCE=<int>      -> pin that GPU (explicit manual override).
    3. Pre-set CUDA_VISIBLE_DEVICES -> HONORED, never overridden. This is the
       SLURM path: with gres (gpu/shard) + cgroup ConstrainDevices, each job is
       confined to its allocated card, which it sees as cuda:0. Independent
       instances therefore spread across GPUs on their own.
       (nvidia-smi ignores CUDA_VISIBLE_DEVICES, so auto-picking here would
        clobber the allocation - hence we must not.)
    4. auto / gpu                   -> pick a GPU ourselves: deterministic spread
       by SLURM_LOCALID when present, else the freest GPU per nvidia-smi.

MEMORY BEHAVIOR (backend-specific, applied by configureDevices())
    TensorFlow : enable memory growth (grabs all VRAM otherwise).
    PyTorch    : nothing needed (allocator is already incremental);
                 enables TF32 matmuls (Ampere/Blackwell+) as a bonus.
    JAX        : disable XLA preallocation (~75% by default) - set at import.
'''

import os
import numpy as np

# Policy: auto (default) | cpu | gpu | <int>. Overridable via env var.
DEVICE_PREFERENCE = os.environ.get("DEVICE_PREFERENCE", "auto").lower()

# When we pick the GPU ourselves and more than one is visible (and there is no
# SLURM task hint), choose the one with the most free memory (via nvidia-smi).
AUTO_PICK_FREEST_GPU = True

# Guard so selection runs exactly once even if imported/called multiple times.
_SELECTION_DONE = False


def _queryGPUFreeMem():
    # Free memory (MiB) per PHYSICAL GPU via nvidia-smi; [] if none / fails.
    # Note: nvidia-smi reports all physical GPUs and ignores
    # CUDA_VISIBLE_DEVICES - which is why we only call this when no external
    # binding (e.g. SLURM) is already in place. Under a cgroup that constrains
    # devices, nvidia-smi itself only sees the allocated card, so this
    # correctly returns a single entry there.
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"], encoding="utf-8")
        return [int(x) for x in out.strip().splitlines() if x.strip()]
    except Exception:
        return []


def _select_device():
    #**********************************************************
    # Backend-agnostic device SELECTION via environment vars.
    # Runs once, at import, BEFORE any framework touches CUDA.
    #**********************************************************
    global _SELECTION_DONE
    if _SELECTION_DONE:
        return
    _SELECTION_DONE = True

    backend = os.environ.get("KERAS_BACKEND", "tensorflow").lower()
    print(f"\n  [DataML_Backend] backend='{backend}', preference='{DEVICE_PREFERENCE}'")

    # JAX preallocates ~75% of GPU memory by default; disabling it is the
    # closest analog to TF's memory growth. Must be set before JAX initializes,
    # so we do it here at selection (import) time rather than later.
    if backend == "jax" and DEVICE_PREFERENCE != "cpu":
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    # 1. Explicit CPU.
    if DEVICE_PREFERENCE == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        print("  -> CPU forced by preference.\n")
        return

    # 2. Explicit GPU index (manual override; wins over auto/SLURM).
    if DEVICE_PREFERENCE.isdigit():
        os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE_PREFERENCE
        print(f"  -> Pinned to GPU {DEVICE_PREFERENCE} (explicit preference).")
        return

    # 3. Honor an externally-set CUDA_VISIBLE_DEVICES (SLURM --gres, or manual).
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:  # present and non-empty
        print(f"  -> Honoring pre-set CUDA_VISIBLE_DEVICES='{cvd}' "
              f"(SLURM/externally managed).")
        return

    # 4. Auto: choose a GPU ourselves.
    free = _queryGPUFreeMem()
    if not free:
        if DEVICE_PREFERENCE == "gpu":
            print("  -> GPU requested but none detected; using CPU.\n")
        else:
            print("  -> No GPU detected; running on CPU.\n")
        return

    n = len(free)
    localid = os.environ.get("SLURM_LOCALID", os.environ.get("SLURM_PROCID"))
    if n == 1:
        idx = 0
        print("  -> Single GPU visible; using it.")
    elif localid is not None:
        # SLURM launched tasks but did NOT bind a GPU per task. Spread them
        # deterministically: task k -> GPU (k mod n). Avoids the race where
        # simultaneously-started tasks all see the same 'freest' GPU.
        idx = int(localid) % n
        print(f"  -> SLURM task (localid={localid}) -> GPU {idx} of {n}.")
    elif AUTO_PICK_FREEST_GPU:
        idx = int(np.argmax(free))
        print(f"  -> Auto-picked freest GPU {idx} of {n}.")
    else:
        idx = 0
        print(f"  -> Using GPU 0 of {n}.")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)


def configureDevices():
    #**********************************************************
    # Apply the backend-specific MEMORY behavior. Call from
    # main(); the device itself was already selected at import.
    #**********************************************************
    _select_device()  # no-op if already done at import; safety net
    backend = os.environ.get("KERAS_BACKEND", "tensorflow").lower()
    if backend == "tensorflow":
        _configureTF()
    elif backend == "torch":
        _configureTorch()
    elif backend == "jax":
        _configureJAX()
    else:
        print(f"  (No device hook for backend '{backend}'; relying on Keras defaults.)\n")


def _configureTF():
    # TensorFlow grabs all VRAM at startup unless memory growth is enabled.
    try:
        import tensorflow as tf
    except Exception as e:
        print(f"  [TF] device config skipped (TensorFlow unavailable): {e}\n")
        return
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("  [TF] no visible GPU - running on CPU.\n")
        return
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError as e:
            # Raised if the GPUs were already initialized before this ran.
            print(f"  [TF] could not set memory growth (already initialized?): {e}")
    print(f"  [TF] {len(gpus)} GPU(s) visible, memory growth enabled.\n")


def _configureTorch():
    # PyTorch's caching allocator is already incremental, so there is no
    # 'memory growth' to configure. Enable TF32 matmuls (Ampere/Blackwell+).
    try:
        import torch
    except Exception as e:
        print(f"  [torch] device config skipped (PyTorch unavailable): {e}\n")
        return
    if not torch.cuda.is_available():
        print("  [torch] no CUDA device visible - running on CPU.\n")
        return
    try:
        torch.set_float32_matmul_precision("high")  # TF32 where supported
    except Exception:
        pass
    print(f"  [torch] GPU: {torch.cuda.get_device_name(0)} "
          f"(cuda:0 of {torch.cuda.device_count()}); allocator is incremental.\n")


def _configureJAX():
    # Preallocation was disabled at selection/import time (must precede JAX
    # init); nothing more to do here but report.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    print("  [jax] preallocation disabled; Keras/JAX will manage device placement.\n")


# Perform device SELECTION immediately on import, i.e. BEFORE the importing
# script's own `import keras`. This is the whole point of the module.
_select_device()
