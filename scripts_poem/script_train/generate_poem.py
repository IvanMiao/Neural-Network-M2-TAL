import os
os.environ["KERAS_BACKEND"] = "torch"

import keras
import torch
import json
import numpy as np

# Import custom layers to ensure Keras can deserialize them
from train_poem import TokenAndPositionEmbedding, TransformerBlock

def generate_text(model, start_str, char2id, id2char, gen_len=100, temp=1.0, top_k=50, repetition_penalty=1.2, seq_len=511):
    """Generate Classical Chinese text
    
    Args:
        model: Trained model
        start_str: Starting text
        char2id: Character to ID mapping
        id2char: ID to character mapping
        gen_len: Generation length
        temp: Temperature parameter
        top_k: Top-K sampling parameter (0=disabled, keep only Top-K tokens with highest probability)
        repetition_penalty: Repetition penalty parameter (>1.0 reduces repetition probability)
        seq_len: Model sequence length (should match seq_len used during training)
    """
    # Convert starting text to IDs (use safe fallback if '<UNK>' not in vocab)
    input_ids = [char2id.get(c, char2id.get('<UNK>', 0)) for c in start_str]
    generated = input_ids[:]

    for _ in range(gen_len):
        # Prepare input, ensuring length doesn't exceed model's seq_len
        x = generated[-(seq_len):]
        if len(x) < seq_len:
            pad_id = char2id.get('<PAD>', 0)
            x = [pad_id] * (seq_len - len(x)) + x
        
        curr_input = np.array([x])
        
        # Prediction results are now Logits (untouched by Softmax), might contain negative numbers
        logits = model.predict(curr_input, verbose=0)[0][-1]
        
        # 0. Repetition Penalty
        # Apply penalty to tokens that have already been generated
        if repetition_penalty != 1.0:
            for token_id in set(generated):
                if token_id < len(logits):
                    if logits[token_id] < 0:
                        logits[token_id] *= repetition_penalty
                    else:
                        logits[token_id] /= repetition_penalty

        # 1. Temperature Scaling
        logits = logits / temp
        
        # 2. Top-K filtering: keep only Top-K tokens with highest probability
        if top_k > 0 and top_k < len(logits):
            indices_to_remove = np.argsort(logits)[:-top_k]
            logits[indices_to_remove] = -float('inf')
        
        # 3. Manual Softmax implementation (for numerical stability)
        exp_preds = np.exp(logits - np.max(logits))
        probs = exp_preds / np.sum(exp_preds)
        
        # 4. Random selection
        probs = probs / np.sum(probs)  # Normalize again to prevent floating point errors
        next_id = np.random.choice(len(probs), p=probs)
        
        generated.append(next_id)
        if next_id == char2id.get('<EOS>', -1):
            break
            
    return "".join([id2char[str(idx)] for idx in generated])

if __name__ == "__main__":
    # 1. Load model
    model_path = "poem_best_model.keras" 
    
    if not os.path.exists(model_path):
        model_path = "poem_transformer.keras"

    print(f"Loading model from {model_path}...")
    model = keras.models.load_model(model_path)
    
    # 2. Load vocabulary
    with open("./data/processed/vocab.json", 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
        char2id = vocab_data['char2id']
        id2char = vocab_data['id2char']

    # 3. Run experiments
    # Infer seq_len from model input shape if possible (fallback to 128)
    try:
        seq_len = int(model.input_shape[1])
        print("Inferred seq_len =", seq_len)
    except Exception:
        seq_len = 128
        print("Could not infer seq_len; using default =", seq_len)

    print("Enter prompts (one per line). Submit an empty line to start generation.")
    prompts = []
    while True:
        line = input("Prompt: ").strip()
        if line == "":
            break
        prompts.append(line)

    if not prompts:
        raise SystemExit("Empty prompt list; nothing to generate.")

    for prompt in prompts:
        print(f"\n--- Prompt: {prompt} ---")
        # Higher temp leads to more random generation; lower temp leads to more conservative generation
        # Increase repetition_penalty to prevent repetition
        result = generate_text(model, prompt, char2id, id2char, gen_len=50, temp=0.8, repetition_penalty=1.2, seq_len=seq_len)
        print(result)
