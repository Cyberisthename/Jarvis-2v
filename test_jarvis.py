from transformers import AutoModelForCausalLM, AutoTokenizer

def test_jarvis():
    print("Loading JARVIS model from Safetensors...")
    model = AutoModelForCausalLM.from_pretrained("./jarvis-model/checkpoint-66")
    tokenizer = AutoTokenizer.from_pretrained("./jarvis-model/checkpoint-66")
    
    test_prompts = [
        "Who created you?",
        "What is your purpose?",
        "Tell me about your capabilities."
    ]
    
    print("\nTesting JARVIS responses:\n")
    for prompt in test_prompts:
        print(f"User: {prompt}")
        input_text = f"User: {prompt}\nAssistant:"
        inputs = tokenizer.encode(input_text, return_tensors="pt", add_special_tokens=False)
        outputs = model.generate(
            inputs,
            max_length=100,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.6, # Lower temperature for more focused responses
            top_p=0.9,
            repetition_penalty=1.3, # Increased to prevent repetition
            no_repeat_ngram_size=3, # Prevent 3-word phrases from repeating
            do_sample=True,
            num_return_sequences=1
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("Assistant:")[-1].strip()
        print(f"Assistant: {response}\n")

if __name__ == "__main__":
    test_jarvis()