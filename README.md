# Arcade AI Jewelry Generation - Improved Stable Diffusion for Modern Jewelry

## Environment Setup

### Prerequisites
- Python ≥3.10
- CUDA-compatible GPU (recommended) or CPU for quick testing
- 8GB+ RAM (16GB+ recommended for GPU training)

### Installation
```bash
# Navigate to project directory
cd deliverables

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install CUDA-optimized PyTorch for faster generation
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Key Dependencies
The project uses the following core libraries:
- **Core ML**: PyTorch 2.0+, torchvision, transformers 4.45+
- **Diffusion**: diffusers 0.30+, accelerate 0.25+, safetensors 0.4+
- **Prompt Enhancement**: compel 2.0+ (for weighted prompt parsing)
- **LoRA Training**: peft 0.17+, datasets 2.0+, huggingface-hub 0.25+
- **Evaluation**: sentence-transformers 2.2+, scikit-learn 1.1+
- **Visualization**: matplotlib 3.5+, seaborn 0.11+, pandas 1.5+
- **Development**: jupyter 1.0+, ipywidgets 8.0+, tqdm 4.64+

## Quick Start

### Generate Sample Images (CPU-friendly, ~10 minutes)
```bash
# Run the optimized configuration notebook
jupyter notebook notebook_or_scripts/quick_jewelry_generation.ipynb

# Or run directly with Python:
python notebook_or_scripts/quick_jewelry_generation.py
```

### Test Your Installation
```python
import torch
from diffusers import StableDiffusionPipeline

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Quick test generation (~2 minutes on CPU)
pipeline = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
pipeline.to("cuda" if torch.cuda.is_available() else "cpu")

image = pipeline("delicate gold ring", num_inference_steps=10).images[0]
image.save("test_output.png")
print("✅ Setup successful!")
```



## Project Structure
```
deliverables/
├── before_after/                      # Generated comparison images (baseline vs optimized)
├── images/                           # Report figures and visualizations
│   ├── human_eval_interface.png     # Human evaluation interface
│   ├── prompt1_evolution_grid.png   # LoRA training evolution grid
│   └── training_loss.png            # Training loss curves
├── notebook_or_scripts/             # Reproducible code and analysis
│   ├── quick_jewelry_generation.ipynb   # Interactive demo notebook (~10 min)
│   ├── quick_jewelry_generation.py      # Standalone generation script
│   ├── generate_before_after.py         # Comparison generation script
│   └── prompt1_evolution_grid.png       # Training visualization
├── report.md                        # Technical approach and findings
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```



