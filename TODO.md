# TODO: Fix Jarvis and Train on GPU RTX 5050

## Completed
- [x] Analyzed project: Jarvis chatbot using transformers/GGUF models.
- [x] Identified issue: Nonsense responses from transformers model.
- [x] User requested switch to GGUF for legal/custom reasons.
- [x] Deleted transformers models (jarvis-model, jarvis-model-updated).
- [x] Updated app.py to use llama-cpp-python with jarvis-7b-q4_0.gguf.
- [x] Installing llama-cpp-python with CUDA support.

## Pending
- [ ] Wait for llama-cpp-python installation to complete.
- [ ] Run the Flask app: `python app.py`
- [ ] Test Jarvis responses via browser at localhost:5000.
- [ ] If still nonsense, adjust generation params (lower temp, add stop words).
- [ ] Optionally, retrain model if GGUF needs fine-tuning (use train scripts with GPU).
- [ ] Verify GPU usage with nvidia-smi.

## Notes
- GGUF model: jarvis-7b-q4_0.gguf (assumed custom).
- GPU: RTX 5050 configured in app.py (n_gpu_layers=-1).
- If issues, check console logs or adjust n_ctx.
