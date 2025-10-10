"""
J.A.R.V.I.S. Training Script
============================
This script will train your custom J.A.R.V.I.S. model on your local machine.
It's designed to be as simple as possible while still creating a good model.
"""

import os
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import json
import gc

print("🤖 J.A.R.V.I.S. Training System")
print("==============================")

# --- Configure GPU ---
print("\n🔍 Checking your system...")

# Configure CUDA for RTX 5050
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

try:
    if torch.cuda.is_available():
        # Enable TF32 for better performance on RTX cards
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(0)
        
        # Clean up GPU memory
        torch.cuda.empty_cache()
        gc.collect()
        
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        device = torch.device("cuda")
    else:
        print("⚠️ No GPU detected, falling back to CPU (this will be slow)")
        device = torch.device("cpu")
except Exception as e:
    print(f"⚠️ GPU initialization warning: {str(e)}")
    print("Continuing with GPU anyway...")
    device = torch.device("cuda")

# --- Configuration ---
print("\n⚙️ Setting up training configuration...")
with open("train_config_new.json", "r") as f:
    config = json.load(f)

# --- Load Training Data ---
print("\n📚 Loading training data...")
with open("training-data/custom_jarvis_data.txt", "r", encoding="utf-8") as f:
    training_data = f.read().split("\n\n")

# Process the data into a format suitable for training
dataset_dict = {"text": training_data}
dataset = Dataset.from_dict(dataset_dict)

# --- Set up the Model ---
print("\n🚀 Setting up the model...")
model = AutoModelForCausalLM.from_pretrained("gpt2")
model = model.to(device)  # Move model to GPU
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

print("✅ Model loaded successfully!")

# --- Process Training Data ---
print("\n🔄 Processing training data...")
def tokenize_function(examples):
    return tokenizer(examples["text"], padding=True, truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# --- Configure Training ---
print("\n⚡ Configuring training...")
training_args = TrainingArguments(
    output_dir=config["model_path"],
    num_train_epochs=config["num_train_epochs"],
    per_device_train_batch_size=config["train_batch_size"],
    gradient_accumulation_steps=config["gradient_accumulation_steps"],
    learning_rate=config["learning_rate"],
    fp16=config["fp16"],
    save_steps=config["save_steps"],
    max_grad_norm=config["max_grad_norm"],
    warmup_steps=config["warmup_steps"],
    weight_decay=config["weight_decay"],
    adam_epsilon=config["adam_epsilon"],
    no_cuda=False  # Force GPU usage
)

# Set up the trainer
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# --- Start Training ---
print("\n🎯 Starting training process...")
print("This might take a while. You'll see progress updates here.")
print("It's normal if it seems slow - the model is learning!")

try:
    trainer.train()
    print("\n💾 Saving your custom J.A.R.V.I.S. model...")
    trainer.save_model()
    print("\n✅ Training complete! Your model is ready!")
    print(f"📁 Model saved in: {config['model_path']}")
except Exception as e:
    print(f"\n❌ An error occurred during training: {str(e)}")
    print("\nTry closing other programs and running again.")