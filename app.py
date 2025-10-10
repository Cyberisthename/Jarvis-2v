from flask import Flask, render_template, request, jsonify
from llama_cpp import Llama

app = Flask(__name__)

# Load the GGUF model globally
print("Loading Jarvis GGUF model...")
llm = Llama(
    model_path="jarvis-7b-q4_0.gguf",
    n_gpu_layers=-1,  # Use CPU for now, change to -1 for GPU once CUDA build works
    n_ctx=2048,  # Context length
    verbose=False
)
print("Model loaded successfully!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('message', '')

    # Format input like training data
    prompt = f"User: {user_input}\nAssistant:"

    # Generate response using GGUF model
    output = llm(
        prompt,
        max_tokens=512,
        temperature=0.7,
        top_p=0.9,
        repeat_penalty=1.1,  # Reduce repetition
        stop=["User:", "\n"],  # Stop at next user message or newline
        echo=False
    )

    response = output['choices'][0]['text'].strip()

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True, port=5000)