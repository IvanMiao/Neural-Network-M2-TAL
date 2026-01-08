import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
import torch
import json

from eval_callback import TrainingReportCallback

# 1. Load data and vocabulary -> Moved to __main__ to support dual-stream config

# 2. Transformer Block
@keras.saving.register_keras_serializable()
class TransformerBlock(keras.layers.Layer):
    """Pre-LN Transformer Block - More stable training characteristics"""
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
            dropout=rate  # Attention dropout for regularization
        )
        # FFN with dropout between layers
        self.ffn = keras.Sequential([
            keras.layers.Dense(ff_dim, activation="gelu"),  # [TODO] GELU vs ReLU ?
            keras.layers.Dropout(rate),  # FFN internal dropout
            keras.layers.Dense(embed_dim),
        ])
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    def call(self, inputs, training=False):
        # Pre-LN: normalization before processing for more stable training
        x = self.layernorm1(inputs)
        attn_output = self.att(x, x, use_causal_mask=True, training=training)
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

# Positional embedding layer to address issues with PyTorch graph reuse
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
        # Embedding scaling: prevent embedding values from being overwhelmed by positional embeddings
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

def build_xy_from_seqs(seqs, max_len=256, pad_id=0, bos_id=1, eos_id=2):
    kept = []
    for s in seqs:
        s2 = [bos_id] + list(s) + [eos_id]
        if len(s2) <= max_len:
            kept.append(s2)

    N = len(kept)
    data = torch.full((N, max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(kept):
        data[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    X = data[:, :-1]
    y = data[:, 1:]
    return X, y

@keras.saving.register_keras_serializable()
class Perplexity(keras.metrics.Metric):
    def __init__(self, name='perplexity', **kwargs):
        super().__init__(name=name, **kwargs)
        self.acc_loss = self.add_weight(name="acc_loss", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")

    def update_state(self, y_true, y_pred, sample_weight=None):
        losses = self.loss_fn(y_true, y_pred)
        # 创建 mask：忽略 PAD token (PAD id = 0)
        mask = keras.ops.cast(y_true != 0, "float32")
        # 只计算非 PAD token 的 loss
        masked_losses = losses * mask
        self.acc_loss.assign_add(keras.ops.sum(masked_losses))
        self.count.assign_add(keras.ops.sum(mask))

    def result(self):
        return keras.ops.exp(self.acc_loss / (self.count + 1e-6))

    def reset_state(self):
        self.acc_loss.assign(0.0)
        self.count.assign(0.0)

def build_model(vocab_size, seq_len, embed_dim=128, num_heads=4, num_layers=4, dropout_rate=0.2):
    """Build Pre-LN GPT-style Transformer model
    
    Args:
        vocab_size: Vocabulary size
        seq_len: Sequence length
        embed_dim: Embedding dimensions (overfitting -> reduce it )
        num_heads: Number of attention heads
        num_layers: Number of Transformer layers (overfitting -> reduce it)
        dropout_rate: Dropout rate (overfitting -> augment it)
    """
    inputs = keras.Input(shape=(seq_len,))
    
    # Use custom layers for Embedding, including scaling and Dropout
    x = TokenAndPositionEmbedding(seq_len, vocab_size, embed_dim, dropout_rate)(inputs)
    
    for _ in range(num_layers):
        x = TransformerBlock(embed_dim, num_heads, embed_dim * 4, rate=dropout_rate)(x)
    
    # Pre-LN requires a final LayerNorm layer
    x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Output layer doesn't use softmax; set from_logits=True in loss for better numerical stability
    outputs = keras.layers.Dense(vocab_size)(x)
    return keras.Model(inputs, outputs)

if __name__ == "__main__":
    # ================= CONFIGURATION =================
    # Choose mode: "prose" or "poetry"
    TRAIN_MODE = "prose" 
    
    BATCH_SIZE = 64
    EPOCHS = 30
    # =================================================

    print(f"=== Starting training in [{TRAIN_MODE}] mode ===")
    
    # Setup paths based on mode
    if TRAIN_MODE == "prose":
        data_path = "./data/processed/train_prose.pt"
        model_save_name = "prose_model.keras"
    elif TRAIN_MODE == "poetry":
        data_path = "./data/processed/train_poetry.pt"
        model_save_name = "poetry_model.keras"
    else:
        raise ValueError(f"Unknown TRAIN_MODE: {TRAIN_MODE}")

    # Load Vocab (Shared)
    vocab_path = "./data/processed/vocab.json"
    
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
        char2id = vocab["char2id"]
        vocab_size = len(char2id)
        print(f"Vocab size: {vocab_size}")
    
    pad_id = char2id.get('<PAD>', 0)
    bos_id = char2id.get('<BOS>', 1)
    eos_id = char2id.get('<EOS>', 2)
    max_len = 512

    # Load Data
    print(f"Loading data from {data_path}...")
    data = torch.load(data_path, map_location="cpu")
    
    # Check if data is a Tensor and has sufficient length to be considered "pre-windowed"
    # Matches logic in train_poem.py to differentiate between sliding window tensors and raw sequences
    if hasattr(data, 'ndim') and data.ndim == 2 and data.shape[1] > (max_len - 2):
        print("Detected fixed-window dataset. Using simple slicing.")
        X, y = data[:, :-1], data[:, 1:]
    else:
        print("Detected variable-length sequences or short windows. Building X,y via build_xy_from_seqs().")
        # Ensure data is a list of lists or similar iterable
        # If data is a tensor, iterating it yields rows, which works for build_xy_from_seqs
        X, y = build_xy_from_seqs(data, max_len=max_len, pad_id=pad_id, bos_id=bos_id, eos_id=eos_id)
        
    seq_len = X.shape[1]

    # Build Model
    model = build_model(vocab_size, seq_len)
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=5e-4, weight_decay=0.01),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy", Perplexity()]
    )

    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="best_model.keras",
            save_best_only=True,
            monitor="val_loss",
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        TrainingReportCallback(output_dir="./reports", model_name=TRAIN_MODE)
    ]

    print(f"Starting training on {X.shape[0]} samples...")
    # Add shuffle=True for better training quality and add validation set
    model.fit(
        X, y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        shuffle=True,
        validation_split=0.1
    )

    # Save the final model after training
    model.save(model_save_name)
    print(f"Final model saved to {model_save_name}")
