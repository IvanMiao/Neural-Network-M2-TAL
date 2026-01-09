import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
import json
import numpy as np
import sys

# Add project root to sys.path to allow importing from src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
src_dir = os.path.join(project_root, 'src')

if project_root not in sys.path:
    sys.path.append(project_root)
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Import custom layers from train modules/scripts
from src.scripts_prose.train.train import TokenAndPositionEmbedding
from src.scripts_prose.train.train import TransformerBlock as GeneratorBlock
from src.scripts_prose.train.train_classifier import TransformerBlock as ClassifierBlock
# Import generation logic
from src.inference.generate import generate_text

def load_classifier(model_path):
    print(f"Loading classifier from {model_path}...")
    custom_objects = {
        "TokenAndPositionEmbedding": TokenAndPositionEmbedding,
        "TransformerBlock": ClassifierBlock
    }
    return keras.models.load_model(model_path, custom_objects=custom_objects)

def load_generator(model_path):
    print(f"Loading generator from {model_path}...")
    custom_objects = {
        "TokenAndPositionEmbedding": TokenAndPositionEmbedding,
        "TransformerBlock": GeneratorBlock
    }
    return keras.models.load_model(model_path, custom_objects=custom_objects)

def classify_prompt(classifier, prompt, char2id, max_len=20):
    """
    Classify prompt as Prose (0) or Poetry (1).
    """
    # Convert text to IDs
    input_ids = [char2id.get(c, char2id.get('<UNK>', 1)) for c in prompt]
    
    # Truncate or Pad to fit classifier input shape
    # Classifier input is fixed length (e.g. 20)
    seq_len = classifier.input_shape[1] 
    
    x = input_ids[:seq_len] if len(input_ids) > seq_len else input_ids
    pad_id = char2id.get('<PAD>', 0)
    if len(x) < seq_len:
        x = x + [pad_id] * (seq_len - len(x))
        
    curr_input = np.array([x])
    
    # Predict: Output is sigmoid probability of Class 1 (Poetry)
    prob_poetry = classifier.predict(curr_input, verbose=0)[0][0]
    
    predicted_class = 1 if prob_poetry > 0.5 else 0
    confidence = prob_poetry if predicted_class == 1 else 1 - prob_poetry
    
    return predicted_class, confidence

def generate_text_pipeline(classifier, prose_model, poetry_model, prompt, char2id, id2char):
    # 1. Classify
    style_id, confidence = classify_prompt(classifier, prompt, char2id)
    style_name = "Poetry" if style_id == 1 else "Prose"
    print(f"  -> Classification: {style_name} (Confidence: {confidence:.2f})")
    
    # 2. Select Model
    model = poetry_model if style_id == 1 else prose_model
    
    # 3. Generate params
    if style_id == 1: # Poetry
        gen_len = 100
        temp = 0.9
        rep_penalty = 1.8
    else: # Prose
        gen_len = 200
        temp = 0.8
        rep_penalty = 1.5
    
    print(f"  -> Generating {style_name}...")
    
    # 4. Generate using imported logic
    result = generate_text(
        model=model, 
        start_str=prompt, 
        char2id=char2id, 
        id2char=id2char, 
        gen_len=gen_len, 
        temp=temp, 
        repetition_penalty=rep_penalty,
        seq_len=model.input_shape[1] # Use model's actual input shape
    )
    
    return result

if __name__ == "__main__":
    # ================= PATHS =================
    CLASSIFIER_PATH = "style_classifier.keras"
    PROSE_MODEL_PATH = "prose_model.keras"
    POETRY_MODEL_PATH = "poetry_model.keras"
    VOCAB_PATH = "./data/processed/vocab.json"
    # =========================================
    
    # Check files
    missing_files = [p for p in [CLASSIFIER_PATH, PROSE_MODEL_PATH, POETRY_MODEL_PATH, VOCAB_PATH] if not os.path.exists(p)]
    if missing_files:
        print(f"Warning: The following files are missing: {missing_files}")
    
    # Load Vocab
    if not os.path.exists(VOCAB_PATH):
        raise FileNotFoundError(f"Vocab file not found at {VOCAB_PATH}")
        
    with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
        char2id = vocab["char2id"]
        id2char = vocab["id2char"]
        
    # Load Models
    try:
        clf = load_classifier(CLASSIFIER_PATH)
        prose_m = load_generator(PROSE_MODEL_PATH)
        poetry_m = load_generator(POETRY_MODEL_PATH)
        
        print("\n=== Classical Chinese Style-Aware Generation Pipeline ===")
        print("Enter a prompt (or 'q' to quit):")
        
        while True:
            prompt = input("> ")
            if prompt.lower() == 'q':
                break
            if not prompt.strip():
                continue
                
            res = generate_text_pipeline(clf, prose_m, poetry_m, prompt, char2id, id2char)
            print(f"\nResult:\n{res}\n")
            
    except Exception as e:
        print(f"\nError: {e}")
        print("Train the models first or place them in the root directory.")
