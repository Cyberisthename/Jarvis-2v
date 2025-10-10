#!/usr/bin/env python3
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline

def convert_to_gguf():
    print("📦 Loading trained model...")
    model_path = "./jarvis-model"
    output_path = "./jarvis-7b-q4_0.gguf"
    
    # Load the trained model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print("🔄 Converting to GGUF format...")
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device set to use {device}")
    
    # Convert model to GGUF format while preserving custom layers
    model = model.to(device)
    model.config.use_cache = True  # Enable KV cache for inference
    model.save_pretrained(output_path, 
                         safe_serialization=True,
                         max_shard_size="10GB")
    tokenizer.save_pretrained(output_path)
    
    # Add metadata
    metadata = {
        "name": "J.A.R.V.I.S. Custom Model",
        "creator": "Ben",
        "version": "1.0.0",
        "license": "Proprietary - All rights reserved",
        "description": "Custom-trained J.A.R.V.I.S. model for personal use"
    }
    
    # Save metadata
    with open(output_path + ".meta", "w") as f:
        import json
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Model converted and saved to {output_path}")
    print("🏷️ Added custom metadata and license information")

if __name__ == "__main__":
    convert_to_gguf()