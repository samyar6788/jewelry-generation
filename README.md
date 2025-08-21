# Arcade AI Jewelry Generation - Quick Setup Guide

## Environment Setup

### Prerequisites
- Python ≥3.10
- CUDA-compatible GPU (recommended)
- 16GB+ RAM

### Installation
```bash
# Clone or download the project
cd arcade-jewelry-generation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install specific versions for stability
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install diffusers==0.30.3 transformers==4.45.0 accelerate==0.25.0
```

### Required Libraries
```txt
torch>=2.0.0
diffusers>=0.30.0
transformers>=4.45.0
accelerate>=0.25.0
compel>=2.0.0
safetensors>=0.4.0
matplotlib>=3.5.0
seaborn>=0.11.0
pandas>=1.5.0
numpy>=1.21.0
Pillow>=9.0.0
```

## Quick Generation

### Generate Sample Images (CPU-friendly, <10 min)
```python
# Run the optimized configuration notebook
jupyter notebook deliverables/notebook_or_scripts/quick_jewelry_generation.ipynb

# Or run directly:
python deliverables/notebook_or_scripts/generate_jewelry_sample.py
```

### Reproduce Before/After Results
```python
# Generate all 8 required prompts with baseline vs optimized
python deliverables/notebook_or_scripts/generate_before_after.py

# Output: deliverables/before_after/prompt01_baseline.png, prompt01_yours.png, etc.
```

## Key Configuration

### Optimal Settings (from human evaluation)
- **Model**: SD 1.5 (runwayml/stable-diffusion-v1-5)
- **Sampler**: Euler Ancestral
- **Strategy**: medium_compel (Compel prompt weighting)
- **CFG Scale**: 9.0
- **Steps**: 20
- **Resolution**: 512x512

### LoRA Models
Fine-tuned LoRA adapters available for:
- **Threader Earrings**: `models/threader_lora_v3.safetensors`
- **Channel-Set Jewelry**: `models/channelset_lora_v2.safetensors`

Usage:
```python
from diffusers import StableDiffusionPipeline
pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipeline.load_lora_weights("models/threader_lora_v3.safetensors")
```

## File Structure
```
deliverables/
├── before_after/           # 24 comparison images (8 prompts × baseline + optimized)
├── notebook_or_scripts/    # Reproducible code and visualizations
├── report.md              # Technical approach and results (800 words)
└── README.md              # This file

notebook_or_scripts/
├── quick_jewelry_generation.ipynb    # <10min CPU demo
├── generate_before_after.py          # Reproduce all comparisons
├── lora_training_analysis.ipynb      # Training visualizations
└── comprehensive_evaluation.ipynb    # Full experiment results
```

## Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce batch size or use CPU: `device="cpu"`
2. **Slow generation**: Ensure CUDA is available: `torch.cuda.is_available()`
3. **Import errors**: Ensure all dependencies installed: `pip install -r requirements.txt`

### Performance Tips
- Use `torch.float16` for faster generation
- Enable memory efficient attention: `pipeline.enable_memory_efficient_attention()`
- Set deterministic seeds for reproducible results

## Validation

### Test Installation
```python
import torch
from diffusers import StableDiffusionPipeline

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Quick test generation (~2 minutes)
pipeline = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)
image = pipeline("delicate gold ring", num_inference_steps=10).images[0]
image.save("test_output.png")
print("✅ Setup successful!")
```

## Support
For questions or issues, refer to the comprehensive notebooks in `notebook_or_scripts/` which include detailed explanations and troubleshooting steps.
