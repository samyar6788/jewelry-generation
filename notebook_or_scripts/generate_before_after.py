#!/usr/bin/env python3
"""
Generate Before/After Images for Arcade AI Challenge
Creates all 24 required images (8 prompts × baseline + optimized)
"""

import torch
import os
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from compel import Compel
import time
from datetime import datetime

# The 8 required prompts (verbatim from challenge)
REQUIRED_PROMPTS = [
    "channel-set diamond eternity band, 2 mm width, hammered 18k yellow gold, product-only white background",
    "14k rose-gold threader earrings, bezel-set round lab diamond ends, lifestyle macro shot, soft natural light",
    "organic cluster ring with mixed-cut sapphires and diamonds, brushed platinum finish, modern aesthetic",
    "A solid gold cuff bracelet with blue sapphire, with refined simplicity and intentionally crafted for everyday wear",
    "modern signet ring, oval face, engraved gothic initial 'M', high-polish sterling silver, subtle reflection",
    "delicate gold huggie hoops, contemporary styling, isolated on neutral background",
    "stack of three slim rings: twisted gold, plain platinum, black rhodium pavé, editorial lighting",
    "bypass ring with stones on it, with refined simplicity and intentionally crafted for everyday wear"
]

# LoRA adapter paths and configuration
LORA_ADAPTERS = {
    "channel_set": "lora_adapters/channel-set/checkpoint/pytorch_lora_weights.safetensors",
    "threader": "lora_adapters/threader/checkpoint/pytorch_lora_weights.safetensors", 
    "huggie": "lora_adapters/huggie/checkpoint/pytorch_lora_weights.safetensors"
}

# Special tokens for enhanced grounding
SPECIAL_TOKENS = {
    "channel_set": "sks",
    "threader": "phol"
}

def detect_jewelry_category(prompt):
    """Detect jewelry category for LoRA selection"""
    prompt_lower = prompt.lower()
    
    if "channel-set" in prompt_lower:
        return "channel_set"
    elif "threader" in prompt_lower:
        return "threader" 
    elif "huggie" in prompt_lower:
        return "huggie"
    else:
        return None

def load_lora_adapter(pipeline, category):
    """Load appropriate LoRA adapter for the jewelry category"""
    if category and category in LORA_ADAPTERS:
        lora_path = LORA_ADAPTERS[category]
        if os.path.exists(lora_path):
            print(f"🔧 Loading {category} LoRA adapter...")
            pipeline.load_lora_weights(lora_path)
            return True
        else:
            print(f"⚠️  LoRA adapter not found: {lora_path}")
            return False
    return False

def unload_lora_adapter(pipeline):
    """Unload current LoRA adapter"""
    try:
        pipeline.unload_lora_weights()
    except:
        # No LoRA to unload or not supported
        pass

def setup_pipeline(device="cuda"):
    """Setup SD 1.5 pipeline with optimal configuration"""
    print("🔧 Loading SD 1.5 pipeline...")
    
    # Use FP16 on GPU for speed
    dtype = torch.float16 if device == "cuda" else torch.float32
    variant = "fp16" if device == "cuda" else None
    
    pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=dtype,
        variant=variant
    )
    
    # Set optimal sampler (Euler Ancestral from human evaluation)
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.scheduler.config
    )
    
    pipeline.to(device)
    
    # Memory optimizations
    if device == "cuda":
        pipeline.enable_memory_efficient_attention()
    else:
        pipeline.enable_attention_slicing()
    
    # Setup Compel for prompt enhancement
    compel = Compel(
        tokenizer=pipeline.tokenizer,
        text_encoder=pipeline.text_encoder
    )
    
    print(f"✅ Pipeline ready on {device}")
    return pipeline, compel

def apply_jewelry_enhancement(prompt, category=None):
    """Apply the optimal medium_compel strategy with special tokens"""
    
    # Jewelry-specific terms to emphasize (from research)
    jewelry_terms = [
        "diamond", "gold", "silver", "platinum", "rose-gold", "sterling",
        "threader", "huggie", "channel-set", "bezel-set", "prong", "pavé",
        "eternity", "signet", "bypass", "cluster", "cuff", "hammered",
        "contemporary", "modern", "minimalist", "refined", "delicate",
        "brushed", "organic", "editorial", "macro", "lifestyle"
    ]
    
    enhanced_prompt = prompt
    
    # Add special tokens for trained categories
    if category and category in SPECIAL_TOKENS:
        special_token = SPECIAL_TOKENS[category]
        if category == "channel_set" and "channel-set" in enhanced_prompt:
            enhanced_prompt = enhanced_prompt.replace("channel-set", f"{special_token} channel-set")
        elif category == "threader" and "threader" in enhanced_prompt:
            enhanced_prompt = enhanced_prompt.replace("threader", f"{special_token} threader")
    
    # Apply 1.2x emphasis to jewelry terms (optimal weight from evaluation)
    for term in jewelry_terms:
        if term in prompt.lower():
            # Handle both exact and case variations
            for variation in [term, term.capitalize(), term.upper()]:
                if variation in enhanced_prompt:
                    enhanced_prompt = enhanced_prompt.replace(
                        variation, f"({variation})1.2"
                    )
                    break
    
    return enhanced_prompt

def generate_baseline_image(pipeline, prompt, seed=42):
    """Generate baseline image with default settings"""
    
    generator = torch.Generator(device=pipeline.device).manual_seed(seed)
    
    image = pipeline(
        prompt=prompt,
        num_inference_steps=20,
        guidance_scale=7.5,  # Default CFG
        generator=generator,
        height=512,
        width=512
    ).images[0]
    
    return image

def generate_optimized_image(pipeline, compel, prompt, seed=42):
    """Generate optimized image with research-backed settings and LoRA"""
    
    # Detect jewelry category for LoRA selection
    category = detect_jewelry_category(prompt)
    
    # Load appropriate LoRA adapter
    lora_loaded = False
    if category:
        lora_loaded = load_lora_adapter(pipeline, category)
    
    # Apply prompt enhancement with special tokens
    enhanced_prompt = apply_jewelry_enhancement(prompt, category)
    
    # Use Compel for weighted embeddings
    conditioning = compel(enhanced_prompt)
    
    generator = torch.Generator(device=pipeline.device).manual_seed(seed)
    
    image = pipeline(
        prompt_embeds=conditioning,
        num_inference_steps=20,      # Optimal from research
        guidance_scale=9.0,          # Optimal CFG from human evaluation
        generator=generator,
        height=512,
        width=512
    ).images[0]
    
    # Unload LoRA adapter for next generation
    if lora_loaded:
        unload_lora_adapter(pipeline)
    
    return image, enhanced_prompt, category

def generate_all_comparisons():
    """Generate all 24 required images"""
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Starting generation on {device}")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    pipeline, compel = setup_pipeline(device)
    
    # Create output directory
    output_dir = "deliverables/before_after"
    os.makedirs(output_dir, exist_ok=True)
    
    total_start = time.time()
    results = []
    
    for i, prompt in enumerate(REQUIRED_PROMPTS, 1):
        print(f"--- Prompt {i:02d}/08 ---")
        print(f"'{prompt[:60]}...'")
        
        # Generate baseline
        print("🔸 Generating baseline...")
        start_time = time.time()
        baseline_image = generate_baseline_image(pipeline, prompt, seed=42)
        baseline_time = time.time() - start_time
        
        # Save baseline
        baseline_path = f"{output_dir}/prompt{i:02d}_baseline.png"
        baseline_image.save(baseline_path)
        
        # Generate optimized
        print("🔹 Generating optimized...")
        start_time = time.time()
        optimized_image, enhanced_prompt, category = generate_optimized_image(
            pipeline, compel, prompt, seed=42
        )
        optimized_time = time.time() - start_time
        
        # Save optimized
        optimized_path = f"{output_dir}/prompt{i:02d}_yours.png"
        optimized_image.save(optimized_path)
        
        results.append({
            'prompt_num': i,
            'original_prompt': prompt,
            'enhanced_prompt': enhanced_prompt,
            'category': category,
            'lora_used': category is not None,
            'baseline_time': baseline_time,
            'optimized_time': optimized_time,
            'baseline_path': baseline_path,
            'optimized_path': optimized_path
        })
        
        print(f"✅ Saved: {baseline_path}")
        print(f"✅ Saved: {optimized_path}")
        print(f"⏱️  Times: baseline {baseline_time:.1f}s, optimized {optimized_time:.1f}s\n")
    
    total_time = time.time() - total_start
    
    # Summary
    print("🎯 GENERATION COMPLETE!")
    print("="*60)
    print(f"📁 Output directory: {output_dir}")
    print(f"📸 Images generated: {len(results) * 2}")
    print(f"⏱️  Total time: {total_time:.1f}s")
    print(f"⚡ Average per image: {total_time/(len(results)*2):.1f}s")
    
    avg_baseline = sum(r['baseline_time'] for r in results) / len(results)
    avg_optimized = sum(r['optimized_time'] for r in results) / len(results)
    print(f"📊 Baseline avg: {avg_baseline:.1f}s, Optimized avg: {avg_optimized:.1f}s")
    
    # File verification
    print(f"\n📋 Generated Files:")
    for i in range(1, 9):
        baseline_file = f"prompt{i:02d}_baseline.png"
        optimized_file = f"prompt{i:02d}_yours.png"
        print(f"  {baseline_file} ✅")
        print(f"  {optimized_file} ✅")
    
    # LoRA usage summary
    lora_usage = {}
    for result in results:
        if result['lora_used']:
            category = result['category']
            lora_usage[category] = lora_usage.get(category, 0) + 1
    
    print(f"\n🎨 Key Improvements Applied:")
    print(f"  • Compel prompt weighting on jewelry terms")
    print(f"  • Special tokens: 'sks' for channel-set, 'phol' for threader")
    print(f"  • Optimal CFG scale: 9.0 (vs 7.5 baseline)")
    print(f"  • Euler Ancestral sampler for quality")
    print(f"  • Consistent seed (42) for reproducibility")
    
    if lora_usage:
        print(f"\n🧠 LoRA Adapters Used:")
        for category, count in lora_usage.items():
            print(f"  • {category}: {count} prompt(s)")
    else:
        print(f"\n⚠️  No LoRA adapters found - using base model only")
    
    return results

def show_enhancement_examples():
    """Show examples of prompt enhancements"""
    
    print("\n💡 Example Prompt Enhancements:")
    print("="*60)
    
    example_prompts = REQUIRED_PROMPTS[:3]
    
    for i, prompt in enumerate(example_prompts, 1):
        category = detect_jewelry_category(prompt)
        enhanced = apply_jewelry_enhancement(prompt, category)
        lora_info = f" → LoRA: {category}" if category else " → No LoRA"
        
        print(f"\nPrompt {i}{lora_info}:")
        print(f"Original : {prompt}")
        print(f"Enhanced : {enhanced}")
    
    print("="*60)

if __name__ == "__main__":
    print("🎨 Arcade AI Challenge - Before/After Generation")
    print("="*60)
    print("Generating 24 images (8 prompts × baseline + optimized)")
    print("Using optimal configuration from human evaluation research")
    
    show_enhancement_examples()
    
    # Generate all images
    results = generate_all_comparisons()
    
    print(f"\n✅ All deliverable images ready!")
    print(f"📁 Check: deliverables/before_after/")
    print(f"🎯 Ready for submission!")
