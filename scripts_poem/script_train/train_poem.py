import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
import torch
import json


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

def build_xy_from_seqs(seqs, max_len=210, pad_id=0, bos_id=1, eos_id=2, eos_weight=2.0):
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
    w = (y != pad_id).to(torch.float32)
    if eos_weight is not None and eos_weight != 1.0:
        w = w * torch.where(
            y == eos_id,
            torch.tensor(eos_weight, dtype=torch.float32),
            torch.tensor(1.0, dtype=torch.float32),
        )
    return X, y, w

def build_model(vocab_size, seq_len, embed_dim=128, num_heads=4, num_layers=4, dropout_rate=0.2):
    """Build Pre-LN GPT-style Transformer model

    Args:
        vocab_size: Vocabulary size
        seq_len: Sequence length
        embed_dim: Embedding dimensions (reduced to prevent overfitting)
        num_heads: Number of attention heads
        num_layers: Number of Transformer layers (reduced to prevent overfitting)
        dropout_rate: Dropout rate (increased to prevent overfitting)
    """
    inputs = keras.Input(shape=(seq_len,))
    x = TokenAndPositionEmbedding(seq_len, vocab_size, embed_dim, dropout_rate)(inputs)

    for _ in range(num_layers):
        x = TransformerBlock(embed_dim, num_heads, embed_dim * 4, rate=dropout_rate)(x)
    # Pre-LN requires a final LayerNorm layer
    x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Output layer doesn't use softmax; set from_logits=True in loss for better numerical stability
    outputs = keras.layers.Dense(vocab_size)(x)
    return keras.Model(inputs, outputs)

if __name__ == "__main__":
    # 3. Training configuration
    BATCH_SIZE = 64
    EPOCHS = 5
    MODEL_PATH = "poem_transformer.keras"

    with open("./data/processed/vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)
        vocab_size = len(vocab["char2id"])
        char2id = vocab["char2id"]
    seqs = torch.load("./data/processed/train_data.pt")
    max_len = 210
    pad_id = char2id.get('<PAD>', 0)
    bos_id = char2id.get('<BOS>', 1)
    eos_id = char2id.get('<EOS>', 2)

    # txt2trainpt: windows length = seq_len + 1
    if hasattr(seqs, 'ndim') and seqs.ndim == 2 and seqs.shape[1] > (max_len - 2):
        print("Detected fixed-window dataset (likely from txt2trainpt). Using sliding-window X/y from data file.")
        X = seqs[:, :-1]
        y = seqs[:, 1:]
        seq_len = X.shape[1]
        w = (y != pad_id).to(torch.float32)
    else:
        print("Detected variable-length sequences (likely from prepare_trainset). Building X,y via build_xy_from_seqs().")
        X, y, w = build_xy_from_seqs(seqs, max_len=max_len, pad_id=pad_id, bos_id=bos_id, eos_id=eos_id, eos_weight=2.0)
        seq_len = X.shape[1]

    model = build_model(vocab_size=vocab_size, seq_len=seq_len)
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=5e-4, weight_decay=0.01),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="poem_best_model.keras",
            save_best_only=True,
            monitor="val_loss",
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
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
    # Save the final model after training (could be best weights restored from EarlyStopping or weights from the last epoch). 
    # If EarlyStopping is triggered with restore_best_weights=True, the saved model will have the best weights. 
    # For clarity, best_model.keras is a snapshot from the epoch with the lowest val_loss during training.
    model.save(MODEL_PATH)
    print(f"Final model saved to {MODEL_PATH}")
