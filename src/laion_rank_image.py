import sys
import os

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import open_clip
from os.path import expanduser  # pylint: disable=import-outside-toplevel
from urllib.request import urlretrieve  # pylint: disable=import-outside-toplevel
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from CLIP import clip

class LAIONAesthetic():
    def __init__(self, device, clip_model='ViT-L/14'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        if clip_model != "ViT-L/14" and clip_model != "ViT-B/32":
            raise ValueError("Unsupported clip model: " + clip_model)

        self.clip_model_name = clip_model

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

        if clip_model == "ViT-L/14":
            clip_model_path_name = "vit_l_14"
        elif clip_model == "ViT-B/32":
            clip_model_path_name = "vit_b_32"
        else:
            raise ValueError("Unsupported clip model: " + clip_model)

        home = expanduser("~")
        cache_folder = home + "/.cache/emb_reader"
        path_to_model = cache_folder + "/sa_0_4_" + clip_model_path_name + "_linear.pth"
        if not os.path.exists(path_to_model):
            os.makedirs(cache_folder, exist_ok=True)
            url_model = (
                "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_" + clip_model_path_name + "_linear.pth?raw=true"
            )
            urlretrieve(url_model, path_to_model)
        if clip_model_path_name == "vit_l_14":
            m = nn.Linear(768, 1)
        elif clip_model_path_name == "vit_b_32":
            m = nn.Linear(512, 1)
        else:
            raise ValueError("Unsupported clip model: " + clip_model)
        s = torch.load(path_to_model)
        m.load_state_dict(s)
        m.eval()
        m.to(device)
        self.model = m

    def preprocess_tensor_diff(self, img_tensor):
        """
        Differentiable preprocessing pipeline for tensor input.
        Expects an image tensor with shape (C, H, W) and values in [0, 1].
        """
        # Get the original size
        _, H, W = img_tensor.shape

        # Compute scale factor so that the smaller side becomes 224.
        scale = 224 / min(H, W)
        new_H, new_W = int(round(H * scale)), int(round(W * scale))

        # Add a batch dimension and resize using differentiable bicubic interpolation.
        img_tensor = img_tensor.unsqueeze(0)  # Shape: (1, C, H, W)
        resized = F.interpolate(img_tensor, size=(new_H, new_W), mode='bicubic', align_corners=False, antialias=True)

        # Center crop to 224 x 224.
        top = (new_H - 224) // 2
        left = (new_W - 224) // 2
        cropped = resized[:, :, top:top+224, left:left+224].squeeze(0)  # Shape: (C, 224, 224)

        # Normalize in a differentiable way.
        mean = torch.tensor(self.mean, device=cropped.device, dtype=cropped.dtype).view(-1, 1, 1)
        std = torch.tensor(self.std, device=cropped.device, dtype=cropped.dtype).view(-1, 1, 1)
        normalized = (cropped - mean) / std
        return normalized

    def preprocess_image(self, image):
        """
        Unified preprocessing: if a PIL image is provided use non-differentiable PIL-based pipeline;
        if a tensor is provided, use the differentiable pipeline.
        """
        if isinstance(image, torch.Tensor):
            return self.preprocess_tensor_diff(image)
        else:
            return self.preprocess_pil(image).to(self.device)

    def predict_from_pil(self, img):
        img_tensor = self.preprocess_image(img)
        clip_image_embed = F.normalize(
            self.clip_model.encode_image(img_tensor[None, ...]).float(),
            dim=-1)
        score = self.model(clip_image_embed)
        return score

    def predict_from_tensor(self, img_tensor, data_type=torch.float32):
        img_tensor = self.preprocess_image(img_tensor)
        clip_image_embed = F.normalize(
            self.clip_model.encode_image(img_tensor[None, ...]).float(),
            dim=-1)
        score = self.model(clip_image_embed).to(data_type)
        return score