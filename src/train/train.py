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
class TransformerBlock(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            keras.layers.Dense(ff_dim, activation="gelu"),
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

# 新增：位置编码层，解决 PyTorch 图重用问题
class TokenAndPositionEmbedding(keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_emb = keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = keras.ops.shape(x)[-1]
        positions = keras.ops.arange(start=0, stop=maxlen, step=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

def build_model(vocab_size, seq_len, embed_dim=256, num_heads=8, num_layers=4):
    inputs = keras.Input(shape=(seq_len,))
    
    # 使用自定义层处理 Embedding
    x = TokenAndPositionEmbedding(seq_len, vocab_size, embed_dim)(inputs)
    
    for _ in range(num_layers):
        x = TransformerBlock(embed_dim, num_heads, embed_dim * 4)(x)
    
    outputs = keras.layers.Dense(vocab_size, activation="softmax")(x)
    return keras.Model(inputs, outputs)

# 3. 训练配置
model = build_model(vocab_size, seq_len)
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=5e-4, weight_decay=0.01),
    loss="sparse_categorical_crossentropy"
)

callbacks = [
    keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2),
    keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
]

print(f"Starting training on {X.shape[0]} samples...")
# 增加 shuffle=True 提高训练质量
model.fit(X, y, batch_size=64, epochs=20, callbacks=callbacks, shuffle=True)

model.save("shiji_transformer.keras")
print("Model saved successfully.")