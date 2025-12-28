import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
import torch
import json

# 1. 加载数据与词表
data = torch.load("./data/processed/train_data.pt") 
with open("./data/processed/vocab.json", 'r', encoding='utf-8') as f:
    vocab_size = len(json.load(f)['char2id'])

X, y = data[:, :-1], data[:, 1:]
seq_len = X.shape[1]

# 2. Transformer 组件
@keras.saving.register_keras_serializable()
class TransformerBlock(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            keras.layers.Dense(ff_dim, activation="relu"),
            keras.layers.Dense(embed_dim),
        ])
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = keras.layers.Dropout(rate)

    def call(self, inputs, training=False):
        # 使用 use_causal_mask=True 实现 GPT 式单向注意力
        attn_output = self.att(inputs, inputs, use_causal_mask=True, training=training)
        out1 = self.layernorm1(inputs + self.dropout(attn_output, training=training))
        return self.layernorm2(out1 + self.dropout(self.ffn(out1), training=training))

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
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = keras.ops.shape(x)[-1]
        positions = keras.ops.arange(start=0, stop=maxlen, step=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def get_config(self):
        config = super().get_config()
        config.update({
            "maxlen": self.maxlen,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
        })
        return config

def build_model(vocab_size, seq_len, embed_dim=256, num_heads=8, num_layers=4):
    inputs = keras.Input(shape=(seq_len,))
    
    # 使用自定义层处理 Embedding
    x = TokenAndPositionEmbedding(seq_len, vocab_size, embed_dim)(inputs)
    
    for _ in range(num_layers):
        x = TransformerBlock(embed_dim, num_heads, embed_dim * 4)(x)
    
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