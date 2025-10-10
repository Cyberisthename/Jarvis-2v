from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def generate_response(prompt):
    # Load the updated model
    model = AutoModelForCausalLM.from_pretrained("jarvis-model-updated")
    tokenizer = AutoTokenizer.from_pretrained("jarvis-model-updated")
    
    # Format the prompt
    full_prompt = f"User: {prompt}\nAssistant:"
    
    # Generate response
    inputs = tokenizer(full_prompt, return_tensors="pt", padding=True)
    attention_mask = torch.ones_like(inputs.input_ids)
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=attention_mask,
        max_length=500,
        num_return_sequences=1,
        do_sample=True,
        top_p=0.9,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("Assistant:", 1)[1].strip()

# Test some topics JARVIS learned about
test_questions = [
    "What is artificial intelligence?",
    "Can you explain machine learning?",
    "What are the main applications of AI?",
]

print("🧠 Testing JARVIS's New Knowledge:")
print("=" * 40)

for question in test_questions:
    print(f"\nUser: {question}")
    response = generate_response(question)
    print(f"JARVIS: {response}")
    print("-" * 40)