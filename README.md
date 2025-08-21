# Arcade AI Jewelry Generation

A focused pipeline for generating high-quality jewelry images using Stable Diffusion 1.5 with LoRA fine-tuning.

## Environment Setup

### Prerequisites
- Python ≥3.10
- 8GB+ RAM (16GB+ recommended for GPU)

### Installation
```bash
# Clone and navigate to project
cd deliverables

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Demo: Threader Earrings Generation
```bash
# Launch Jupyter and open the threader earrings demo
jupyter notebook notebook_or_scripts/threader_earrings_demo.ipynb
```

This notebook demonstrates the complete optimization pipeline:
- Baseline SD 1.5 generation
- LoRA adapter loading for threader earrings
- Enhanced prompt with special tokens and attention weighting
- Side-by-side comparison

### Test Installation
```python
import torch
from diffusers import StableDiffusionPipeline

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Quick test (~2 minutes)
pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
image = pipeline("gold ring", num_inference_steps=10).images[0]
image.save("test_output.png")
print("✅ Setup successful!")
```

## Project Structure
```
deliverables/
├── before_after/                           # Generated comparison images
├── lora_adapters/                          # Fine-tuned LoRA weights
│   ├── channel-set/checkpoint/            # Channel-set earrings LoRA  
│   └── threader/checkpoint/               # Threader earrings LoRA
├── notebook_or_scripts/                    # Demo notebooks and scripts
│   └── threader_earrings_demo.ipynb      # Main demo notebook
├── report.md                              # Technical findings
└── requirements.txt                       # Dependencies
```



