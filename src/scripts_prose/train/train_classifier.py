import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
import torch
import json
import numpy as np
import unicodedata
import random

# 1. Load data and vocabulary -> Moved to __main__

# 2. Transformer Block
@keras.saving.register_keras_serializable()
class TransformerBlock(keras.layers.Layer):
    """Pre-LN Transformer Block - Adapted for optional causal masking"""
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        # Attention with dropout
        self.att = keras.layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=embed_dim,
            dropout=rate
        )
        # FFN with dropout between layers
        self.ffn = keras.Sequential([
            keras.layers.Dense(ff_dim, activation="gelu"),
            keras.layers.Dropout(rate),
            keras.layers.Dense(embed_dim),
        ])
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    def call(self, inputs, training=False):
        # Pre-LN: normalization before processing
        x = self.layernorm1(inputs)
        # Classifier uses bidirectional attention (no causal mask)
        attn_output = self.att(x, x, use_causal_mask=False, training=training)
        out1 = inputs + self.dropout1(attn_output, training=training)
        
        x = self.layernorm2(out1)
        ffn_output = self.ffn(x, training=training)
        return out1 + self.dropout2(ffn_output, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config

# Positional embedding layer
@keras.saving.register_keras_serializable()
class TokenAndPositionEmbedding(keras.layers.Layer):
    """Token + Position Embedding with sqrt(embed_dim) scaling"""
    def __init__(self, maxlen, vocab_size, embed_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate
        self.embed_scale = keras.ops.sqrt(keras.ops.cast(embed_dim, "float32"))
        self.token_emb = keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
        self.dropout = keras.layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        maxlen = keras.ops.shape(x)[-1]
        positions = keras.ops.arange(start=0, stop=maxlen, step=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x) * self.embed_scale
        return self.dropout(x + positions, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "maxlen": self.maxlen,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "dropout_rate": self.dropout_rate,
        })
        return config

def build_classifier(vocab_size, seq_len, embed_dim=128, num_heads=4, num_layers=4, dropout_rate=0.2):
    """Build BERT-style Transformer Classifier"""
    inputs = keras.Input(shape=(seq_len,), dtype="int32")
    
    # 1. Embedding
    x = TokenAndPositionEmbedding(seq_len, vocab_size, embed_dim, dropout_rate)(inputs)
    
    # 2. Transformer Encoder Blocks (Bidirectional => use_causal_mask=False)
    for _ in range(num_layers):
        x = TransformerBlock(embed_dim, num_heads, embed_dim * 4, rate=dropout_rate)(x)
    
    # 3. Aggregation (Global Average Pooling) to handle variable effective lengths
    x = keras.layers.GlobalAveragePooling1D()(x)
    
    x = keras.layers.Dropout(dropout_rate)(x)
    
    # 4. Classification Head
    outputs = keras.layers.Dense(1, activation="sigmoid")(x) # Binary classification: 0=Prose, 1=Poetry
    
    return keras.Model(inputs, outputs)

def get_punctuation_ids(char2id):
    """Identify punctuation IDs from vocabulary using unicodedata."""
    punct_ids = set()
    for char, idx in char2id.items():
        if len(char) == 1 and unicodedata.category(char).startswith('P'):
            punct_ids.add(idx)
        if char in ["，", "。", "！", "？", "：", "；", "、", "「", "」", "『", "』", "（", "）", "—", "…"]:
            punct_ids.add(idx)
    return punct_ids

def sample_clean_segments(data_tensor, n_samples, punct_ids, min_len=2, max_len=20, pad_id=0):
    """
    Randomly sample segments of length min_len to max_len that DO NOT contain punctuation.
    """
    samples = []
    # each row is a separate document/sequence.
    
    rows = data_tensor.tolist()
    total_generated = 0
    row_indices = list(range(len(rows)))
    
    print(f"Sampling {n_samples} segments from {len(rows)} source sequences...")
    
    while len(samples) < n_samples:
        # Pick random row
        row = rows[random.choice(row_indices)]
        
        # Filter out padding from row end
        try:
            # Find first PAD (0)
            real_len = row.index(pad_id)
        except ValueError:
            real_len = len(row)
            
        if real_len < min_len:
            continue
            
        # Pick random length
        segment_len = random.randint(min_len, min(max_len, real_len))
        
        # Pick random start
        if real_len - segment_len <= 0:
            start_idx = 0
        else:
            start_idx = random.randint(0, real_len - segment_len)
            
        segment = row[start_idx : start_idx + segment_len]
        
        # Check for punctuation
        has_punct = any(token in punct_ids for token in segment)
        
        if not has_punct:
            # Pad to max_len for batching
            padded_segment = segment + [pad_id] * (max_len - len(segment))
            samples.append(padded_segment)
            total_generated += 1
            if total_generated % 5000 == 0:
                print(f"  Generated {total_generated}/{n_samples}")
                
    return torch.tensor(samples, dtype=torch.long)

if __name__ == "__main__":
    # ================= CONFIGURATION =================
    BATCH_SIZE = 64
    EPOCHS = 10 
    VALIDATION_SPLIT = 0.1
    SAMPLES_PER_CLASS = 20000 
    MIN_LEN = 2
    MAX_LEN = 20
    # =================================================

    print(f"=== Starting CLASSIFIER training ===")
    
    # Setup paths
    prose_path = "./data/processed/train_prose.pt"
    poetry_path = "./data/processed/train_poetry.pt"
    model_save_name = "style_classifier.keras"

    # Load Vocab (Shared)
    vocab_path = "./data/processed/vocab.json"
    
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
        char2id = vocab["char2id"]
        vocab_size = len(char2id)
        print(f"Vocab size: {vocab_size}")
        
    punct_ids = get_punctuation_ids(char2id)
    print(f"Identified {len(punct_ids)} punctuation tokens.")
    
    # Load Data
    print(f"Loading prose data from {prose_path}...")
    prose_data = torch.load(prose_path, map_location="cpu")
    print(f"Sampling CLEAN prose segments...")
    X_prose = sample_clean_segments(prose_data, SAMPLES_PER_CLASS, punct_ids, MIN_LEN, MAX_LEN)
    y_prose = torch.zeros(X_prose.shape[0], 1)
    
    print(f"Loading poetry data from {poetry_path}...")
    poetry_data = torch.load(poetry_path, map_location="cpu")
    print(f"Sampling CLEAN poetry segments...")
    X_poetry = sample_clean_segments(poetry_data, SAMPLES_PER_CLASS, punct_ids, MIN_LEN, MAX_LEN)
    y_poetry = torch.ones(X_poetry.shape[0], 1)
    
    print(f"Prose samples: {X_prose.shape}")
    print(f"Poetry samples: {X_poetry.shape}")
    
    # Concatenate
    X = torch.cat([X_prose, X_poetry], dim=0)
    y = torch.cat([y_prose, y_poetry], dim=0)
    
    seq_len = MAX_LEN
    print(f"Total training samples: {X.shape[0]}, Fixed Seq Len: {seq_len}")

    # Build Model
    model = build_classifier(vocab_size, seq_len, embed_dim=128, num_heads=4, num_layers=2) # Light model for short prompts
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=0.01),
        loss=keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=["accuracy"]
    )
    
    model.summary()

    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="best_classifier.keras",
            save_best_only=True,
            monitor="val_loss",
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
    ]

    print(f"Starting training...")
    
    # Manual shuffle
    indices = torch.randperm(X.shape[0])
    X = X[indices]
    y = y[indices]

    model.fit(
        X, y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        shuffle=True,
        validation_split=VALIDATION_SPLIT
    )

    # Save the final model
    model.save(model_save_name)
    print(f"Final model saved to {model_save_name}")
