from __future__ import annotations

from pathlib import Path
from typing import Final

from PIL import Image
import torch
from torchvision import transforms

# ---------------------------------------------------------------------------
# Local helper modules
# ---------------------------------------------------------------------------
# Assuming all helper modules live next to this file or are discoverable via
# PYTHONPATH. Adjust the import paths as needed for your project layout.

from src.laion_rank_image import LAIONAesthetic
from src.simulacra_rank_image import SimulacraAesthetic
from src.laion_v2_rank_image import LAIONV2Aesthetic

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
DEVICE: Final[torch.device] = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
CLIP_MODEL_NAME: Final[str] = "ViT-L/14"  # Works for both v1 + v2

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def load_tensor(img_path: Path, device: torch.device) -> torch.Tensor:
    """Load *img_path* → FloatTensor in [0,1] on *device*."""
    pil = Image.open(img_path).convert("RGB")
    return transforms.ToTensor()(pil).to(device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(img_path: str | Path) -> None:  # noqa: D401 – simple function
    img_path = Path(img_path)
    if not img_path.exists():
        raise FileNotFoundError(img_path)

    print(f"[example.py] Evaluating aesthetic scores for: {img_path}")
    pil_image = Image.open(img_path).convert("RGB")
    tensor_image = load_tensor(img_path, DEVICE)

    # Instantiate predictors
    laion = LAIONAesthetic(device=DEVICE, clip_model=CLIP_MODEL_NAME)
    simulacra = SimulacraAesthetic(device=DEVICE)
    laion_v2 = LAIONV2Aesthetic(device=DEVICE, clip_model=CLIP_MODEL_NAME)

    # PIL pipeline (non‑diff)
    laion_score_pil = laion.predict_from_pil(pil_image)
    simulacra_score_pil = simulacra.predict_from_pil(pil_image)
    laion_v2_score_pil = laion_v2.predict_from_pil(pil_image)

    # Tensor pipeline (differentiable)
    laion_score_tensor = laion.predict_from_tensor(tensor_image)
    simulacra_score_tensor = simulacra.predict_from_tensor(tensor_image)
    laion_v2_score_tensor = laion_v2.predict_from_tensor(tensor_image)

    # ------------------------------------------------------------------
    # Output neatly
    # ------------------------------------------------------------------
    print("\n=== Aesthetic scores ===")
    print(f"LAION v1   (PIL):   {laion_score_pil.item():.4f}")
    print(f"LAION v1   (Tensor):{laion_score_tensor.item():.4f}")
    print(f"Simulacra  (PIL):   {simulacra_score_pil.item():.4f}")
    print(f"Simulacra  (Tensor):{simulacra_score_tensor.item():.4f}")
    print(f"LAION v2   (PIL):   {laion_v2_score_pil.item():.4f}")
    print(f"LAION v2   (Tensor):{laion_v2_score_tensor.item():.4f}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python example.py <image_path>")
        print("Example: python example.py test.png")
        print("Using default test image 'test.png'")
        main("test.png")
    else:
        main(sys.argv[1])