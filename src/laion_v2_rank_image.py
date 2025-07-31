from __future__ import annotations

import contextlib
import os
from pathlib import Path
from urllib.request import urlretrieve

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms  # noqa: F401 – kept for symmetry with sister modules

try:
    import clip  # Official OpenAI package
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "The 'clip' package is required – install it with `pip install git+https://github.com/openai/CLIP.git`."
    ) from e

__all__ = ['LAIONV2Aesthetic']


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _l2_normalize(arr: torch.Tensor, axis: int = -1) -> torch.Tensor:
    norm = arr.norm(p=2, dim=axis, keepdim=True).clamp(min=1e-8)  # ← safe
    return arr / norm

def _download_weights(target: Path, variant: str) -> None:
    'Download the pre‑trained MLP weights if they are missing.'
    target.parent.mkdir(parents=True, exist_ok=True)
    base_url = 'https://huggingface.co/camenduru/improved-aesthetic-predictor/resolve/main/'
    url = base_url + variant
    print(f'[improved_rank_image] Downloading weights → {target} …')
    urlretrieve(url, target)


# ---------------------------------------------------------------------------
# Lightweight MLP used by Improved‑Aesthetic‑Predictor
# ---------------------------------------------------------------------------

class _ImprovedMLP(nn.Module):
    'Simple 5‑layer MLP (copied from the reference repo).'

    def __init__(self, embedding_dim: int = 768) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.layers(x)


# ---------------------------------------------------------------------------
# Main convenience wrapper
# ---------------------------------------------------------------------------

class LAIONV2Aesthetic:
    '''Wrapper around the *Improved Aesthetic Predictor* (a.k.a. LAION v2).

    Parameters
    ----------
    device        : torch.device | str, optional – defaults to CUDA if available.
    clip_model    : str, optional – only 'ViT-L/14' is officially supported.
    weight_variant: str, optional – filename of the pre‑trained MLP checkpoint.
    cache_dir     : Path | str, optional – where to store downloaded weights.
    '''

    def __init__(
        self,
        device: str | torch.device | None = None,
        *,
        clip_model: str = 'ViT-L/14',
        weight_variant: str = 'sac+logos+ava1-l14-linearMSE.pth',
        cache_dir: str | Path | None = None,
    ) -> None:
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )

        # 1) CLIP image encoder ------------------------------------------------
        self.clip_model_name = clip_model
        self.clip_model, self._preprocess_pil = clip.load(self.clip_model_name, device=self.device)
        self.clip_model.eval().requires_grad_(False)  # freeze params, but keep autograd alive for inputs

        # 2) MLP head ----------------------------------------------------------
        if cache_dir is None:
            cache_dir = Path(os.path.expanduser('~/.cache/emb_reader'))
        else:
            cache_dir = Path(cache_dir)
        weights_path = cache_dir / weight_variant
        if not weights_path.exists():
            _download_weights(weights_path, weight_variant)

        self.mlp = _ImprovedMLP(embedding_dim=768).to(self.device)
        state_dict = torch.load(weights_path, map_location=self.device)
        self.mlp.load_state_dict(state_dict, strict=True)
        self.mlp.eval().requires_grad_(False)  # we only want grads w.r.t. the *image*, not the weights

        # 3) Normalisation parameters -----------------------------------------
        self._mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device).view(-1, 1, 1)
        self._std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).view(-1, 1, 1)

    # ----------------------------------------------------------------------
    # Pre‑processing utilities
    # ----------------------------------------------------------------------

    @staticmethod
    def _resize_shorter_side(img: torch.Tensor, target: int = 224) -> torch.Tensor:
        'Resize so the shorter side equals *target* (expects C×H×W, values∈[0,1]).'
        _, h, w = img.shape
        scale = target / min(h, w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        img = img.unsqueeze(0)
        img = F.interpolate(img, size=(new_h, new_w), mode='bicubic', align_corners=False, antialias=True)
        return img.squeeze(0)

    def _preprocess_tensor(self, img: torch.Tensor) -> torch.Tensor:
        'Differentiable preprocessing mirroring CLIPʼs PIL pipeline.'
        img = img.to(self.device)
        img = self._resize_shorter_side(img)
        # centre‑crop 224×224
        _, h, w = img.shape
        top, left = (h - 224) // 2, (w - 224) // 2
        img = img[:, top:top + 224, left:left + 224]
        return (img - self._mean) / self._std

    # ----------------------------------------------------------------------
    # Internals
    # ----------------------------------------------------------------------

    def _embed(self, img: torch.Tensor, *, no_grad: bool = True) -> torch.Tensor:
        'Encode *img* with CLIP and L2‑normalise (toggle autograd via *no_grad*).'
        ctx = torch.no_grad() if no_grad else contextlib.nullcontext()
        with ctx:
            emb = self.clip_model.encode_image(img).float()
        return _l2_normalize(emb, axis=-1)

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------

    def predict_from_pil(self, image: Image.Image) -> torch.Tensor:
        'Fast convenience path (non‑differentiable).'
        img_tensor = self._preprocess_pil(image).unsqueeze(0).to(self.device)
        emb = self._embed(img_tensor, no_grad=True)
        with torch.no_grad():
            score = self.mlp(emb)
        return score.squeeze()

    def predict_from_tensor(self, img_tensor: torch.Tensor, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        '''Differentiable aesthetic prediction.

        `img_tensor` must be C×H×W in the range [0,1]. Call `.backward()` on the
        returned scalar to obtain gradients w.r.t. the *input image*.
        '''
        img_tensor = self._preprocess_tensor(img_tensor).unsqueeze(0)
        img_tensor.requires_grad_(True)  # ensure autograd tracks this leaf
        emb = self._embed(img_tensor, no_grad=False)
        score = self.mlp(emb)
        return score.to(dtype).squeeze()
