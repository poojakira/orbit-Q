"""
Custom CUDA / Triton Kernel — Anomaly Score Fusion
Fuses IsolationForest and Autoencoder anomaly scores on the GPU using a
weighted harmonic mean, implemented both as a pure Triton kernel (GPU)
and a NumPy fallback (CPU-only environments).

Usage:
    from orbit_q.engine.kernels.anomaly_fusion import fuse_scores
    fused = fuse_scores(iso_scores, ae_scores, iso_weight=0.6)
"""

import numpy as np
from typing import Optional

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


# ── Triton Kernel (GPU) ───────────────────────────────────────────────────────
if TRITON_AVAILABLE and TORCH_AVAILABLE:

    @triton.jit
    def _fusion_kernel(
        iso_ptr,
        ae_ptr,
        out_ptr,
        iso_w,
        ae_w,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        iso = tl.load(iso_ptr + offsets, mask=mask)
        ae = tl.load(ae_ptr + offsets, mask=mask)

        # Normalize to [0, 1] range using sigmoid
        iso_norm = 1.0 / (1.0 + tl.exp(iso))  # higher score = more nominal
        ae_norm = ae / (ae + 1.0)  # AE: higher = more anomalous → invert
        ae_norm_inv = 1.0 - ae_norm

        # Weighted harmonic mean
        denom = iso_w / iso_norm + ae_w / ae_norm_inv
        fused = (iso_w + ae_w) / denom

        tl.store(out_ptr + offsets, fused, mask=mask)

    def _triton_fuse(iso: np.ndarray, ae: np.ndarray, iso_w: float, ae_w: float) -> np.ndarray:
        device = "cuda"
        iso_t = torch.tensor(iso, dtype=torch.float32, device=device)
        ae_t = torch.tensor(ae, dtype=torch.float32, device=device)
        out = torch.empty_like(iso_t)
        n = iso_t.numel()
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        _fusion_kernel[grid](iso_t, ae_t, out, iso_w, ae_w, n, BLOCK_SIZE=1024)
        return out.cpu().numpy()


# ── NumPy CPU fallback ────────────────────────────────────────────────────────
def _numpy_fuse(iso: np.ndarray, ae: np.ndarray, iso_w: float, ae_w: float) -> np.ndarray:
    """Weighted harmonic mean on CPU using normalized scores."""
    # IsolationForest: higher (less negative) = more nominal
    iso_norm = 1.0 / (1.0 + np.exp(-iso))  # sigmoid → [0,1]
    # AE: higher reconstruction error = more anomalous, invert
    ae_norm = ae / (ae.max() + 1e-9)
    ae_norm_inv = np.clip(1.0 - ae_norm, 1e-9, 1.0)

    denom = iso_w / np.clip(iso_norm, 1e-9, 1.0) + ae_w / ae_norm_inv
    return (iso_w + ae_w) / denom


# ── Public API ────────────────────────────────────────────────────────────────
def fuse_scores(
    iso_scores: np.ndarray,
    ae_scores: np.ndarray,
    iso_weight: float = 0.6,
    ae_weight: Optional[float] = None,
) -> np.ndarray:
    """
    Fuse IsolationForest and Autoencoder anomaly scores.

    Returns an array in [0, 1] where values closer to 0 indicate anomalies
    and values closer to 1 indicate nominal behaviour.

    Uses CUDA Triton kernel when available, falls back to NumPy otherwise.
    """
    ae_w = ae_weight if ae_weight is not None else (1.0 - iso_weight)

    if TRITON_AVAILABLE and TORCH_AVAILABLE and torch.cuda.is_available():
        return _triton_fuse(iso_scores, ae_scores, iso_weight, ae_w)

    return _numpy_fuse(iso_scores, ae_scores, iso_weight, ae_w)


def classify_fused(fused_scores: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert fused scores to -1 (anomaly) / +1 (nominal) labels."""
    return np.where(fused_scores < threshold, -1, 1).astype(int)
