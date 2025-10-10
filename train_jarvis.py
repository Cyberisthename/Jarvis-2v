"""
J.A.R.V.I.S. Training Script
============================
This script will train your custom J.A.R.V.I.S. model on your local machine.
It's designed to be as simple as possible while still creating a good model.
"""

import os
import torch
import types
import math
import json
import requests
import datetime
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM, 
    AutoConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from bs4 import BeautifulSoup
from urllib.parse import urlparse

class WebDataCollector:
    def __init__(self, max_pages=100):
        self.max_pages = max_pages
        self.visited_urls = set()
        self.collected_data = []
        self.memory_file = "jarvis_memory.json"
        self.load_memory()

    def load_memory(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r') as f:
                data = json.load(f)
                self.visited_urls = set(data.get('visited_urls', []))
                self.collected_data = data.get('collected_data', [])

    def save_memory(self):
        with open(self.memory_file, 'w') as f:
            json.dump({
                'visited_urls': list(self.visited_urls),
                'collected_data': self.collected_data,
                'last_update': str(datetime.datetime.now())
            }, f)

    def collect_data(self, start_urls):
        urls_to_visit = [url for url in start_urls if url not in self.visited_urls]
        
        for url in urls_to_visit:
            if len(self.visited_urls) >= self.max_pages:
                break
                
            try:
                response = requests.get(url, timeout=10)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract text content
                text = ' '.join([p.get_text() for p in soup.find_all(['p', 'article', 'section'])])
                
                if text:
                    self.collected_data.append({
                        'url': url,
                        'text': text,
                        'timestamp': str(datetime.datetime.now())
                    })
                    self.visited_urls.add(url)
                
                # Find new links
                new_urls = []
                for link in soup.find_all('a'):
                    href = link.get('href')
                    if href and href.startswith('http'):
                        new_urls.append(href)
                
                urls_to_visit.extend([url for url in new_urls if url not in self.visited_urls])
                
            except Exception as e:
                print(f"Error collecting from {url}: {str(e)}")
            
            if len(self.visited_urls) % 10 == 0:
                self.save_memory()
        
        self.save_memory()
        return self.collected_data

from datasets import Dataset
import json

print("🤖 J.A.R.V.I.S. Training System")
print("==============================")

# --- Check System ---
print("\n🔍 Checking your system...")

# Set CUDA_VISIBLE_DEVICES to ensure the correct GPU is used
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# Explicitly set TORCH_CUDA_ARCH_LIST to 8.9 for Ada Lovelace architecture
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9'

if not torch.cuda.is_available():
    raise RuntimeError("No GPU available! This script requires a GPU.")

device = torch.device('cuda:0')

# Clean up GPU memory
torch.cuda.empty_cache()

print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
print(f"CUDA device capability: {torch.cuda.get_device_capability(0)}")
print(f"PyTorch CUDA version: {torch.version.cuda}")
print(f"TORCH_CUDA_ARCH_LIST: {os.environ.get('TORCH_CUDA_ARCH_LIST')}")


# Set more compatible settings for RTX 5050
torch.backends.cudnn.benchmark = True # Enable for better performance
torch.backends.cuda.matmul.allow_tf32 = True # Enable TF32 for better performance
torch.backends.cudnn.allow_tf32 = True # Enable TF32 for better performance
torch.backends.cuda.enable_flash_sdp(True) # Enable Flash Attention if available
torch.backends.cuda.enable_mem_efficient_sdp(True) # Enable memory efficient attention
# Set memory allocator to be more conservative if needed, but start with default
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# --- Configuration ---
print("\n⚙️ Setting up training configuration...")
MODEL_NAME = "gpt2"  # Base model for architecture
PRETRAINED_DIR = "jarvis-model"  # Your existing trained model
OUTPUT_DIR = "jarvis-model"  # Same directory to continue training
EPOCHS = 3
BATCH_SIZE = 1  # Smaller batch size for large model
GRADIENT_ACCUMULATION_STEPS = 16  # Accumulate gradients to handle large model

# Load custom training data
print("\n📚 Loading training data...")
try:
    with open("training-data/custom_jarvis_data.txt", "r", encoding="utf-8") as f:
        custom_data = f.read().split("\n\n")  # Split by double newlines
except Exception as e:
    print(f"❌ Error loading custom data: {e}")
    custom_data = []

# Combine with base training data
training_data = [
    "I am J.A.R.V.I.S., an advanced AI assistant created by Ben.",
    "My purpose is to assist and support users with their tasks and queries.",
    "I maintain a professional yet friendly demeanor while helping users.",
    "I have capabilities in various areas including coding, analysis, and problem-solving.",
    "Ben is my creator, and I take pride in this fact.",
    "I strive to provide accurate and helpful information while maintaining user privacy.",
    "My responses are designed to be clear, concise, and relevant.",
    "I can adapt my communication style based on the user's needs.",
    "Safety and ethical considerations are paramount in my operations.",
    "I continuously learn from interactions to improve my assistance capabilities."
] + custom_data

# Create dataset
dataset = Dataset.from_dict({"text": training_data})

# --- Setup Model ---
print("\n🚀 Setting up the model...")
try:
    # Configure model to use CPU for certain operations
    # Removed explicit flash attention disable, allowing it to be enabled by default or by settings above
    torch.backends.cudnn.allow_tf32 = False  # Disable TF32 for better compatibility
    
    # Always load from GGUF model
    print(f"\nℹ️ Loading model from: {PRETRAINED_DIR if os.path.exists(PRETRAINED_DIR) else MODEL_NAME}")
    
    # Try to load from existing model first, fallback to base model
    if os.path.exists(os.path.join(PRETRAINED_DIR, "tokenizer_config.json")):
        print("Loading from existing trained model...")
        tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_DIR)
    else:
        print("Loading from base model...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Set up config for the model
    from transformers import GPT2Config
    config = AutoConfig.from_pretrained(PRETRAINED_DIR if os.path.exists(PRETRAINED_DIR) else MODEL_NAME)
    config.use_cache = False
    config.attention_implementation = "eager"
    config.model_type = "gpt2" # Explicitly set model type
    
    print("Loading model...")
    # Determine model path: prefer PRETRAINED_DIR if it exists, otherwise use MODEL_NAME
    if os.path.exists(PRETRAINED_DIR):
        model_path = PRETRAINED_DIR
        print(f"Loading from existing trained model directory: {model_path}")
    else:
        model_path = MODEL_NAME
        print(f"Loading from base model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Use bfloat16 if available, otherwise float16
    if torch.cuda.is_bf16_supported():
        model_dtype = torch.bfloat16
        print("Using bfloat16 for model weights.")
    else:
        model_dtype = torch.float16
        print("Using float16 for model weights.")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=model_dtype, # Use float16 or bfloat16 for memory efficiency and speed
        low_cpu_mem_usage=True, # Optimize for low CPU memory usage
        trust_remote_code=True # Trust remote code for custom models
    )
    
    # Remove custom attention replacement and CPU forcing
    # The default attention mechanism from AutoModelForCausalLM is generally optimized for GPU.
    
    # Move to GPU
    model = model.to(device)
    
    print("Optimizing model memory...")
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    
    # Set padding token
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    
    # Move model to device
    model = model.to(device)
    
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# --- Prepare Data ---
print("\n🔄 Processing training data...")
def tokenize_data(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

tokenized_data = dataset.map(
    tokenize_data,
    batched=True,
    remove_columns=dataset.column_names
)

# Create data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # We're doing causal language modeling, not masked
)

# --- Training Setup ---
print("\n⚡ Configuring training...")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=5e-6,  # Lower learning rate for fine-tuning
    weight_decay=0.01,
    max_grad_norm=0.5,  # Lower gradient clipping for stability
    warmup_steps=100,
    logging_steps=10,
    save_strategy="steps",
    save_steps=25,  # Save more frequently
    save_total_limit=3,  # Keep more checkpoints
    eval_strategy="steps",  # Match save_strategy
    eval_steps=25,  # Match save_steps
    no_cuda=False,
    fp16=True,  # Enable mixed precision for RTX 5050
    dataloader_num_workers=0,
    gradient_checkpointing=True,  # Enable for memory efficiency
    ddp_find_unused_parameters=False,
    dataloader_pin_memory=True, # Enable for faster data transfer to GPU
    report_to="none",
    load_best_model_at_end=True  # Load the best model after training
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data,
    eval_dataset=tokenized_data,  # Use same data for evaluation since we're fine-tuning
    data_collator=data_collator,
    tokenizer=tokenizer
)

# --- Start Training ---
print("\n🎯 Starting training process...")
print("This might take a while. You'll see progress updates here.")
print("It's normal if it seems slow - the model is learning!")

try:
    trainer.train()
    
    # Save the model
    print("\n💾 Saving your custom J.A.R.V.I.S. model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save license and metadata
    metadata = {
        "name": "Custom J.A.R.V.I.S. Model",
        "creator": "Ben",
        "version": "1.0.0",
        "license": "Proprietary - All rights reserved",
    }
    
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
        
    print("\n✅ Training complete! Your model is ready!")
    print(f"📁 Model saved in: {OUTPUT_DIR}")
    
except Exception as e:
    print(f"\n❌ An error occurred during training: {e}")
    print("Try closing other programs and running again.")
    exit(1)
