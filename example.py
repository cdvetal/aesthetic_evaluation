from PIL import Image
import torch
from torchvision import transforms
from src.laion_rank_image import LAIONAesthetic
from src.simulacra_rank_image import SimulacraAesthetic

# Load the image
image_path = "test.png"
pil_image = Image.open(image_path)

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Convert the image to a tensor
img_tensor = transforms.ToTensor()(pil_image).to(device)

# Initialize the LAION and Simulacra aesthetic models
laion_model = LAIONAesthetic(device=device, clip_model="vit_l_14")
simulacra_model = SimulacraAesthetic(device=device)

# Predict using PIL image
laion_pil_score = laion_model.predict_from_pil(pil_image)
simulacra_pil_score = simulacra_model.predict_from_pil(pil_image)

# Predict using tensor image
laion_tensor_score = laion_model.predict_from_tensor(img_tensor)
simulacra_tensor_score = simulacra_model.predict_from_tensor(img_tensor)

# Print the results
print(f"LAION model score (PIL): {laion_pil_score.item()}")
print(f"LAION model score (Tensor): {laion_tensor_score.item()}")
print(f"Simulacra model score (PIL): {simulacra_pil_score.item()}")
print(f"Simulacra model score (Tensor): {simulacra_tensor_score.item()}")