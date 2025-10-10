"""
J.A.R.V.I.S. Continuous Learning Script
=====================================
This script allows JARVIS to learn from internet sources while maintaining its core identity.
"""

import os
import torch
import json
import requests
import datetime
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

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
        # Structure the knowledge better
        memory_data = {
            'visited_urls': list(self.visited_urls),
            'collected_data': self.collected_data,
            'last_update': str(datetime.datetime.now()),
            'stats': {
                'total_words_collected': sum(len(data['text'].split()) for data in self.collected_data),
                'total_urls_processed': len(self.visited_urls),
                'knowledge_categories': {
                    'ai_ml': sum(1 for data in self.collected_data if any(kw in data['url'].lower() for kw in ['ai', 'artificial-intelligence', 'machine-learning'])),
                    'programming': sum(1 for data in self.collected_data if any(kw in data['url'].lower() for kw in ['programming', 'python', 'code'])),
                    'research': sum(1 for data in self.collected_data if any(kw in data['url'].lower() for kw in ['research', 'arxiv', 'science']))
                }
            }
        }
        
        with open(self.memory_file, 'w') as f:
            json.dump(memory_data, f, indent=2)

    def find_interesting_links(self, soup, current_url):
        """Find relevant and interesting links to explore"""
        interesting_links = []
        base_domain = urlparse(current_url).netloc
        
        for link in soup.find_all('a'):
            href = link.get('href')
            if not href:
                continue
                
            # Make relative URLs absolute
            if href.startswith('/'):
                href = f"https://{base_domain}{href}"
            elif not href.startswith('http'):
                continue
                
            # Skip social media and irrelevant sites
            skip_domains = ['facebook.com', 'twitter.com', 'instagram.com', 'youtube.com', 'ads']
            if any(domain in href for domain in skip_domains):
                continue
                
            # Prioritize educational and knowledge sites
            priority_domains = ['wikipedia.org', 'github.com', 'stackoverflow.com', 'docs.', '.edu', 'medium.com']
            if any(domain in href for domain in priority_domains):
                interesting_links.insert(0, href)
            else:
                interesting_links.append(href)
                
        return interesting_links

    def collect_data(self, start_urls):
        urls_to_visit = [url for url in start_urls if url not in self.visited_urls]
        
        while urls_to_visit:
            if len(self.visited_urls) >= self.max_pages:
                break
                
            url = urls_to_visit.pop(0)
            try:
                print(f"� Exploring: {url}")
                response = requests.get(url, timeout=10, headers={'User-Agent': 'JARVIS Learning Bot'})
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
                    print(f"✅ Successfully collected {len(text.split())} words")
                
                # Extract title and description for context
                title = soup.title.string if soup.title else ""
                description = ""
                meta_desc = soup.find('meta', {'name': 'description'})
                if meta_desc:
                    description = meta_desc.get('content', '')
                
                # Extract main content
                text = ' '.join([p.get_text() for p in soup.find_all(['p', 'article', 'section'])])
                
                if text:
                    self.collected_data.append({
                        'url': url,
                        'title': title,
                        'description': description,
                        'text': text,
                        'timestamp': str(datetime.datetime.now())
                    })
                    self.visited_urls.add(url)
                    print(f"📚 Learned {len(text.split())} words from {title}")
                
                # Find interesting links to explore next
                new_urls = self.find_interesting_links(soup, url)
                print(f"🔗 Found {len(new_urls)} interesting links to explore")
                
                # Add new URLs to explore
                urls_to_visit.extend([url for url in new_urls if url not in self.visited_urls])
                
            except Exception as e:
                print(f"❌ Error collecting from {url}: {str(e)}")
            
            if len(self.visited_urls) % 10 == 0:
                self.save_memory()
                print(f"💾 Saved progress: {len(self.visited_urls)} URLs processed")
        
        self.save_memory()
        return self.collected_data

def continuous_learn(seed_urls=None, max_pages=1000):
    """Main continuous learning function"""
    print("\n🧠 J.A.R.V.I.S. Web Exploration System")
    print("=" * 40)

    # Starting points for web exploration
    if not seed_urls:
        seed_urls = [
            # Technology and AI
            "https://en.wikipedia.org/wiki/Artificial_intelligence",
            "https://en.wikipedia.org/wiki/Machine_learning",
            # Programming and Development
            "https://stackoverflow.com/questions/tagged/python",
            "https://github.com/topics/artificial-intelligence",
            # Science and Research
            "https://arxiv.org/list/cs.AI/recent",
            "https://www.nature.com/articles/s41467-020-17167-8",
            # Tech News and Updates
            "https://techcrunch.com/artificial-intelligence/",
            "https://www.technologyreview.com/topic/artificial-intelligence/"
        ]
    
    print("🌐 Starting web exploration...")
    print("I'll learn from interesting content while exploring the web.")

    # Initialize collector
    collector = WebDataCollector(max_pages=100)
    
    print("\n🌐 Collecting knowledge from the internet...")
    web_data = collector.collect_data(seed_urls)
    
    print("\n📚 Loading base model for fine-tuning...")
    # Use a permissively licensed model that allows for derivative works.
    # NousResearch/Llama-2-7b-chat-hf is a good candidate with an Apache 2.0 license.
    base_model_name = "NousResearch/Llama-2-7b-chat-hf"
    new_model_name = "jarvis-7b-tuned-v1"

    # Configure quantization to train more efficiently on consumer GPUs
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # Standard device detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quantization_config,
        device_map="auto", # Automatically handle device placement
    )
    model.config.use_cache = False # Recommended for training

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token # Set padding token
    
    # Prepare data for training
    print("\n🔄 Processing collected data...")
    training_texts = []
    for data in web_data:
                # Format as conversation to maintain JARVIS identity
        text = data['text']
        title = data.get('title', '')
        description = data.get('description', '')
        
        # Clean and format the text
        text = text.strip()
        text = ' '.join(text.split())  # Remove extra whitespace
        
        # Create a system context message
        system_context = (
            "You are J.A.R.V.I.S., an advanced AI assistant. You provide accurate, helpful, "
            "and concise responses. You maintain a professional yet friendly tone. "
            "When you're not sure about something, you say so directly."
        )
        
        # Break into smaller chunks and maintain context
        chunks = [text[i:i+512] for i in range(0, len(text), 512)]
        for i, chunk in enumerate(chunks):
            context = f"Topic: {title}\nBackground: {description}\n\n" if i == 0 else "Continuing previous topic...\n\n"
            
            # Format as a structured conversation
            training_texts.append(
                f"System: {system_context}\n\n"
                f"{context}"
                f"User: Please explain this topic clearly and accurately.\n"
                f"Assistant: I'll help you understand this topic. {chunk}\n\n"
                f"User: Can you summarize the key points?\n"
                f"Assistant: Here are the main points to understand: {chunk[:200]}...\n\n"
            )
    
    # Tokenize data
    training_encodings = tokenizer(
        training_texts,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Create Dataset class
    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = item['input_ids'].clone()
            return item

        def __len__(self):
            return len(self.encodings['input_ids'])
    
    # Create train and validation datasets
    total_size = len(training_encodings['input_ids'])
    train_size = int(0.9 * total_size)
    
    train_dataset = TextDataset({
        key: val[:train_size] for key, val in training_encodings.items()
    })
    
    eval_dataset = TextDataset({
        key: val[train_size:] for key, val in training_encodings.items()
    })
    
    print(f"\nDataset stats:")
    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(eval_dataset)}")
    
    # Training arguments optimized for continuous learning
    training_args = TrainingArguments(
        output_dir=new_model_name,
        num_train_epochs=1, # Start with 1 epoch for fine-tuning
        per_device_train_batch_size=2,
        learning_rate=2e-4,
        logging_dir="./logs",
        logging_steps=10,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        weight_decay=0.01,
        fp16=True, # Enable mixed precision training
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        optim="paged_adamw_8bit", # Memory efficient optimizer
        report_to="none",
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    print("\n🚀 Starting continuous learning (fine-tuning)...")
    trainer.train()
    
    print(f"\n💾 Saving updated model to ./{new_model_name}...")
    model.save_pretrained(new_model_name)
    tokenizer.save_pretrained(new_model_name)
    
    print("\n✅ Continuous learning complete!")
    print(f"Your new model has been saved in the '{new_model_name}' directory.")
    print("Next step is to convert it to GGUF format.")

if __name__ == "__main__":
    # Example usage: Add your own URLs to learn from
    urls = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Machine_learning",
        # Add more URLs here
    ]
    continuous_learn(urls)