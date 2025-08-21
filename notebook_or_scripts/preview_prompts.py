#!/usr/bin/env python3
"""
Preview Optimized Prompts
Shows how prompts will be transformed without generating images
"""

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

def apply_jewelry_enhancement(prompt, category=None):
    """Apply custom Compel strategy with specific groupings per prompt"""
    
    enhanced_prompt = prompt
    
    # Add special tokens for trained categories first
    if category and category in SPECIAL_TOKENS:
        special_token = SPECIAL_TOKENS[category]
        if category == "channel_set" and "channel-set" in enhanced_prompt:
            enhanced_prompt = enhanced_prompt.replace("channel-set", f"{special_token} channel-set")
        elif category == "threader" and "threader" in enhanced_prompt:
            enhanced_prompt = enhanced_prompt.replace("threader", f"{special_token} threader")
    
    # Apply prompt-specific Compel groupings
    if "channel-set diamond eternity band" in prompt:
        # Prompt 1: channel-set
        enhanced_prompt = enhanced_prompt.replace("sks channel-set", "(sks channel-set)1.2")
        enhanced_prompt = enhanced_prompt.replace("diamond", "(diamond)1.2")
        enhanced_prompt = enhanced_prompt.replace("hammered", "(hammered)1.2")
        enhanced_prompt = enhanced_prompt.replace("gold", "(gold)1.2")
        enhanced_prompt = enhanced_prompt.replace("product-only", "(product-only)1.2")
        enhanced_prompt = enhanced_prompt.replace("white background", "(white background)1.2")
        
    elif "14k rose-gold threader earrings" in prompt:
        # Prompt 2: threader
        enhanced_prompt = enhanced_prompt.replace("rose-gold", "(rose-gold)1.2")
        enhanced_prompt = enhanced_prompt.replace("phol threader", "(phol threader)1.2")
        enhanced_prompt = enhanced_prompt.replace("bezel-set", "(bezel-set)1.2")
        enhanced_prompt = enhanced_prompt.replace("diamond", "(diamond)1.2")
        enhanced_prompt = enhanced_prompt.replace("lifestyle", "(lifestyle)1.2")
        enhanced_prompt = enhanced_prompt.replace("macro", "(macro)1.2")
        
    elif "organic cluster ring with mixed-cut sapphires" in prompt:
        # Prompt 3: organic cluster
        enhanced_prompt = enhanced_prompt.replace("organic cluster", "(organic cluster)1.2")
        enhanced_prompt = enhanced_prompt.replace("sapphires", "(sapphires)1.2")
        enhanced_prompt = enhanced_prompt.replace("diamonds", "(diamonds)1.2")
        enhanced_prompt = enhanced_prompt.replace("brushed", "(brushed)1.2")
        enhanced_prompt = enhanced_prompt.replace("platinum", "(platinum)1.2")
        enhanced_prompt = enhanced_prompt.replace("modern", "(modern)1.2")
        
    elif "solid gold cuff bracelet with blue sapphire" in prompt:
        # Prompt 4: cuff bracelet
        enhanced_prompt = enhanced_prompt.replace("gold", "(gold)1.2")
        enhanced_prompt = enhanced_prompt.replace("cuff bracelet", "(cuff bracelet)1.2")
        enhanced_prompt = enhanced_prompt.replace("blue sapphire", "(blue sapphire)1.2")
        enhanced_prompt = enhanced_prompt.replace("refined", "(refined)1.2")
        
    elif "modern signet ring, oval face, engraved gothic" in prompt:
        # Prompt 5: signet ring
        enhanced_prompt = enhanced_prompt.replace("modern", "(modern)1.2")
        enhanced_prompt = enhanced_prompt.replace("signet", "(signet)1.2")
        enhanced_prompt = enhanced_prompt.replace("engraved gothic initial 'M'", "(engraved gothic initial 'M')1.2")
        enhanced_prompt = enhanced_prompt.replace("sterling", "(sterling)1.2")
        enhanced_prompt = enhanced_prompt.replace("silver", "(silver)1.2")
        
    elif "delicate gold huggie hoops" in prompt:
        # Prompt 6: huggie hoops
        enhanced_prompt = enhanced_prompt.replace("delicate", "(delicate)1.2")
        enhanced_prompt = enhanced_prompt.replace("gold", "(gold)1.2")
        enhanced_prompt = enhanced_prompt.replace("huggie hoops", "(huggie hoops)1.2")
        enhanced_prompt = enhanced_prompt.replace("contemporary", "(contemporary)1.2")
        
    elif "stack of three slim rings" in prompt:
        # Prompt 7: ring stack
        enhanced_prompt = enhanced_prompt.replace("stack of three", "(stack of three)1.2")
        enhanced_prompt = enhanced_prompt.replace("gold", "(gold)1.2")
        enhanced_prompt = enhanced_prompt.replace("platinum", "(platinum)1.2")
        enhanced_prompt = enhanced_prompt.replace("pavé", "(pavé)1.2")
        enhanced_prompt = enhanced_prompt.replace("editorial", "(editorial)1.2")
        
    elif "bypass ring with stones" in prompt:
        # Prompt 8: bypass ring
        enhanced_prompt = enhanced_prompt.replace("bypass ring", "(bypass ring)1.2")
        enhanced_prompt = enhanced_prompt.replace("refined", "(refined)1.2")
    
    return enhanced_prompt

def preview_all_optimized_prompts():
    """Preview all optimized prompts without generating images"""
    
    print("\n🔍 OPTIMIZED PROMPT PREVIEW")
    print("="*80)
    print("Showing all 8 prompts with optimization transformations")
    print("="*80)
    
    for i, prompt in enumerate(REQUIRED_PROMPTS, 1):
        category = detect_jewelry_category(prompt)
        enhanced = apply_jewelry_enhancement(prompt, category)
        
        # Status indicators
        lora_status = f"✅ LoRA: {category}" if category else "❌ No LoRA"
        special_token = f"🏷️  Token: {SPECIAL_TOKENS.get(category, 'None')}" if category else "🏷️  Token: None"
        
        print(f"\n{'='*15} PROMPT {i:02d}/08 {'='*15}")
        print(f"Category: {lora_status}")
        print(f"Special Token: {special_token}")
        print(f"\n📝 ORIGINAL:")
        print(f"   {prompt}")
        print(f"\n🚀 OPTIMIZED:")
        print(f"   {enhanced}")
        
        # Show key transformations
        changes = []
        if category and category in SPECIAL_TOKENS:
            if category == "channel_set" and "sks channel-set" in enhanced:
                changes.append("Added 'sks' token before 'channel-set'")
            elif category == "threader" and "phol threader" in enhanced:
                changes.append("Added 'phol' token before 'threader'")
        
        # Count emphasis weights
        emphasis_count = enhanced.count(")1.2")
        if emphasis_count > 0:
            changes.append(f"Applied 1.2x emphasis to {emphasis_count} jewelry terms")
        
        if changes:
            print(f"\n🔧 TRANSFORMATIONS:")
            for change in changes:
                print(f"   • {change}")
    
    print(f"\n{'='*80}")
    print(f"📊 SUMMARY:")
    
    # Count categories
    categories = {}
    for prompt in REQUIRED_PROMPTS:
        cat = detect_jewelry_category(prompt)
        if cat:
            categories[cat] = categories.get(cat, 0) + 1
    
    print(f"   • Total prompts: {len(REQUIRED_PROMPTS)}")
    print(f"   • LoRA-enhanced: {sum(categories.values())}")
    print(f"   • Base model only: {len(REQUIRED_PROMPTS) - sum(categories.values())}")
    
    if categories:
        print(f"   • LoRA breakdown:")
        for cat, count in categories.items():
            print(f"     - {cat}: {count} prompt(s)")
    
    print(f"{'='*80}")

if __name__ == "__main__":
    print("🔍 PROMPT OPTIMIZATION PREVIEW")
    print("=" * 50)
    print("Showing all optimized prompts without image generation")
    print("=" * 50)
    
    # Run the preview function
    preview_all_optimized_prompts()
    
    print("\n✅ Preview complete! Run generate_before_after.py to generate images.")
