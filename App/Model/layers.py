from __future__ import annotations

import tensorflow as tf


class IngredientEmbeddingLayer(tf.keras.layers.Layer):
    """Project multi-hot ingredient vectors to dense embeddings."""

    def __init__(self, vocab_size: int, embedding_dim: int = 64, **kwargs) -> None:
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

    def build(self, input_shape):
        self.embedding_table = self.add_weight(
            shape=(self.vocab_size, self.embedding_dim),
            initializer="glorot_uniform",
            trainable=True,
            name="ingredient_embedding_table",
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        weighted = tf.linalg.matmul(inputs, self.embedding_table)
        denominator = tf.reduce_sum(inputs, axis=1, keepdims=True) + 1e-6
        return weighted / denominator

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "embedding_dim": self.embedding_dim,
            }
        )
        return config


class IngredientPositionalEmbedding(tf.keras.layers.Layer):
    """Token embedding with learnable positional embedding."""

    def __init__(self, vocab_size: int, embed_dim: int, max_seq_len: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.token_emb = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            mask_zero=True,
            name="token_embedding",
        )
        self.pos_emb = tf.keras.layers.Embedding(
            input_dim=max_seq_len,
            output_dim=embed_dim,
            name="positional_embedding",
        )

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        del training
        seq_len = tf.shape(x)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        return self.token_emb(x) + self.pos_emb(positions)

    def compute_mask(self, inputs, mask=None):
        del mask
        return self.token_emb.compute_mask(inputs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "max_seq_len": self.max_seq_len,
            }
        )
        return config


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    """Multi-head self-attention built from Dense projections."""

    def __init__(self, embed_dim: int, num_heads: int, **kwargs) -> None:
        super().__init__(**kwargs)
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = embed_dim // num_heads

        self.wq = tf.keras.layers.Dense(embed_dim, use_bias=False, name="Wq")
        self.wk = tf.keras.layers.Dense(embed_dim, use_bias=False, name="Wk")
        self.wv = tf.keras.layers.Dense(embed_dim, use_bias=False, name="Wv")
        self.wo = tf.keras.layers.Dense(embed_dim, use_bias=False, name="Wo")

    def _split_heads(self, x: tf.Tensor, batch_size: tf.Tensor) -> tf.Tensor:
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def _scaled_dot_product_attention(
        self,
        q: tf.Tensor,
        k: tf.Tensor,
        v: tf.Tensor,
        pad_mask: tf.Tensor | None = None,
    ) -> tf.Tensor:
        scores = tf.matmul(q, k, transpose_b=True)
        scores = scores / tf.math.sqrt(tf.cast(self.depth, tf.float32))
        if pad_mask is not None:
            scores += pad_mask * -1e9
        weights = tf.nn.softmax(scores, axis=-1)
        return tf.matmul(weights, v)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        del training
        batch_size = tf.shape(x)[0]

        q = self._split_heads(self.wq(x), batch_size)
        k = self._split_heads(self.wk(x), batch_size)
        v = self._split_heads(self.wv(x), batch_size)

        pad_mask = tf.cast(tf.reduce_sum(tf.abs(x), axis=-1, keepdims=True) == 0.0, tf.float32)
        pad_mask = tf.expand_dims(tf.transpose(pad_mask, perm=[0, 2, 1]), axis=1)

        attn = self._scaled_dot_product_attention(q, k, v, pad_mask)
        attn = tf.transpose(attn, perm=[0, 2, 1, 3])
        attn = tf.reshape(attn, (batch_size, -1, self.embed_dim))
        return self.wo(attn)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
            }
        )
        return config


class TransformerEncoderBlock(tf.keras.layers.Layer):
    """Transformer encoder block with MHSA, FFN, residual, and layer norm."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout_rate: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout_rate = dropout_rate

        self.attention = MultiHeadSelfAttention(embed_dim=embed_dim, num_heads=num_heads, name="mhsa")
        self.dropout_1 = tf.keras.layers.Dropout(dropout_rate)
        self.layernorm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.ffn_dense_1 = tf.keras.layers.Dense(ffn_dim, activation="relu", name="ffn_1")
        self.ffn_dense_2 = tf.keras.layers.Dense(embed_dim, name="ffn_2")
        self.dropout_2 = tf.keras.layers.Dropout(dropout_rate)
        self.layernorm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        attn_out = self.attention(x, training=training)
        attn_out = self.dropout_1(attn_out, training=training)
        x = self.layernorm_1(x + attn_out)

        ffn_out = self.ffn_dense_2(self.ffn_dense_1(x))
        ffn_out = self.dropout_2(ffn_out, training=training)
        return self.layernorm_2(x + ffn_out)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "ffn_dim": self.ffn_dim,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


class AbsDiff(tf.keras.layers.Layer):
    """Element-wise absolute difference between two tensors."""

    def call(self, inputs):
        return tf.abs(inputs[0] - inputs[1])

    def get_config(self):
        return super().get_config()
