# Aesthetic Evaluation Framework

This repository is intended to aggregate several aesthetic evaluation methods into a simple framework.

Each model's interface provides **predict_from_pil** and **predict_from_tensor** methods, allowing both PIL and tensor formats for the input image. The tensor format is particularly useful when gradient calculation is required.

Run example.py for testing the framework.

## Setup

1. **Clone the repository including submodules:**

   If you haven't cloned the repository yet, run:
   
   ```git clone --recursive <repository-url>```

   If you've already cloned the repository without submodules, run:
   
   ```git submodule update --init --recursive```

2. **Create the conda environment:**

    ```conda env create -f environment.yml```

3. **Install openclip:**
    ```python -m pip install open_clip_torch```

## Current Models
- LAION Aesthetic Predictor v1 and v2 
- Simulacra Aesthetic Predictor 

## Usage Example
    python example.py <image_path>
