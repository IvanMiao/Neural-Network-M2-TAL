import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
import torch
import json

# 1. 加载数据与词表
data = torch.load("./data/processed/train_data.pt", map_location="cpu") 
with open("./data/processed/vocab.json", 'r', encoding='utf-8') as f:
    vocab_size = len(json.load(f)['char2id'])

X, y = data[:, :-1], data[:, 1:]
seq_len = X.shape[1]

# 2. Transformer 组件
@keras.saving.register_keras_serializable()
class TransformerBlock(keras.layers.Layer):
    """Pre-LN Transformer Block - 更稳定的训练特性"""
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
            keras.layers.Dense(ff_dim, activation="gelu"),  # GELU 通常优于 ReLU
            keras.layers.Dropout(rate),  # FFN 内部 dropout
            keras.layers.Dense(embed_dim),
        ])
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    def call(self, inputs, training=False):
        # Pre-LN: 先归一化再处理，训练更稳定
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

# 位置编码层，解决 PyTorch 图重用问题
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
        # Embedding 缩放：防止 embedding 值过小被位置编码淹没
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

def build_model(vocab_size, seq_len, embed_dim=128, num_heads=4, num_layers=4, dropout_rate=0.2):
    """构建 Pre-LN GPT 风格的 Transformer 模型
    
    Args:
        vocab_size: 词表大小
        seq_len: 序列长度
        embed_dim: Embedding 维度 (减小以防止过拟合)
        num_heads: 注意力头数
        num_layers: Transformer 层数 (减小以防止过拟合)
        dropout_rate: Dropout 比率 (增加以防止过拟合)
    """
    inputs = keras.Input(shape=(seq_len,))
    
    # 使用自定义层处理 Embedding，包含缩放和 Dropout
    x = TokenAndPositionEmbedding(seq_len, vocab_size, embed_dim, dropout_rate)(inputs)
    
    for _ in range(num_layers):
        x = TransformerBlock(embed_dim, num_heads, embed_dim * 4, rate=dropout_rate)(x)
    
    # Pre-LN 需要最后一层 LayerNorm
    x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    
    # 输出层不使用 softmax，在 loss 中设置 from_logits=True 以提高数值稳定性
    outputs = keras.layers.Dense(vocab_size)(x)
    return keras.Model(inputs, outputs)

if __name__ == "__main__":
    # 3. 训练配置
    BATCH_SIZE = 64
    EPOCHS = 10
    MODEL_PATH = "shiji_transformer.keras"

    model = build_model(vocab_size, seq_len)
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=5e-4, weight_decay=0.01),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath="best_model.keras",
            save_best_only=True,
            monitor="val_loss",
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    ]

    print(f"Starting training on {X.shape[0]} samples...")
    # 增加 shuffle=True 提高训练质量，并添加验证集
    model.fit(
        X, y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        shuffle=True,
        validation_split=0.1
    )

    # 训练结束后保存最终模型（可能是 EarlyStopping 恢复的最佳权重，也可能是最后一个 epoch 的权重）
    # 如果 EarlyStopping 触发且 restore_best_weights=True，这里保存的也是最佳权重
    # 但为了明确区分，best_model.keras 是训练过程中 val_loss 最低时的快照
    model.save(MODEL_PATH)
    print(f"Final model saved to {MODEL_PATH}")