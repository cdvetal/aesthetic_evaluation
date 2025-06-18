import sys
import os

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from PIL import Image
import torch
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import to_pil_image
from simulacra_aesthetic_models.simulacra_fit_linear_model import AestheticMeanPredictionLinearModel
from CLIP import clip

class SimulacraAesthetic():
    def __init__(self, device):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.clip_model_name = 'ViT-B/16'
        self.clip_model = clip.load(self.clip_model_name, jit=False, device=self.device)[0]
        self.clip_model.eval().requires_grad_(False)

        self.mean = [0.48145466, 0.4578275, 0.40821073]
        self.std = [0.26862954, 0.26130258, 0.27577711]
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)

        # Define a unified preprocessing pipeline for PIL images.
        self.preprocess_pil = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            self.normalize,
        ])

        self.model = AestheticMeanPredictionLinearModel(512)
        self.model.load_state_dict(
            torch.load("simulacra_aesthetic_models/models/sac_public_2022_06_29_vit_b_16_linear.pth")
        )
        self.model = self.model.to(self.device)

    def preprocess_tensor_diff(self, img_tensor):
        """
        Differentiable preprocessing pipeline for tensor input.
        Assumes img_tensor shape is (C, H, W) with values in [0,1].
        This function resizes the image preserving aspect ratio (so that the smaller side becomes 224),
        center crops a 224x224 patch, and normalizes the result.
        """
        # Get original size
        _, H, W = img_tensor.shape

        # Compute scale: transform so that the smallest dimension = 224.
        scale = 224 / min(H, W)
        new_H, new_W = int(round(H * scale)), int(round(W * scale))

        # Add a batch dimension and do differentiable resize
        img_tensor = img_tensor.unsqueeze(0)  # shape (1, C, H, W)
        resized = F.interpolate(img_tensor, size=(new_H, new_W), mode='bicubic', align_corners=False, antialias=True)

        # Center crop to 224x224
        top = (new_H - 224) // 2
        left = (new_W - 224) // 2
        cropped = resized[:, :, top:top+224, left:left+224]
        cropped = cropped.squeeze(0)  # shape (C, 224, 224)

        # Normalize, using operations that are differentiable.
        mean = torch.tensor(self.mean, device=cropped.device, dtype=cropped.dtype).view(-1, 1, 1)
        std = torch.tensor(self.std, device=cropped.device, dtype=cropped.dtype).view(-1, 1, 1)
        normalized = (cropped - mean) / std
        return normalized

    def preprocess_image(self, image):
        """
        Unified preprocessing for images coming as PIL images.
        Converts a PIL image to a tensor using a fixed pipeline.
        """
        return self.preprocess_pil(image).to(self.device)

    def predict_from_pil(self, img):
        img_tensor = self.preprocess_image(img)
        clip_image_embed = F.normalize(
            self.clip_model.encode_image(img_tensor[None, ...]).float(),
            dim=-1)
        score = self.model(clip_image_embed)
        return score

    def predict_from_tensor(self, img_tensor, data_type=torch.float32):
        # For a differentiable pipeline, do not convert to PIL.
        img_tensor = self.preprocess_tensor_diff(img_tensor)
        clip_image_embed = F.normalize(
            self.clip_model.encode_image(img_tensor[None, ...]).float(),
            dim=-1)
        score = self.model(clip_image_embed).to(data_type)
        return score