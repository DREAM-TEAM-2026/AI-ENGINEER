from __future__ import annotations

import tensorflow as tf

from app.model.layers import AbsDiff, IngredientPositionalEmbedding, TransformerEncoderBlock


def build_coverage_model(
    vocab_size: int,
    embedding_dim: int = 64,
    num_heads: int = 4,
    ffn_dim: int = 128,
    num_blocks: int = 2,
    max_seq_len: int = 20,
    dropout_rate: float = 0.1,
) -> tf.keras.Model:
    user_input = tf.keras.Input(shape=(max_seq_len,), dtype=tf.int32, name="user_seq")
    recipe_input = tf.keras.Input(shape=(max_seq_len,), dtype=tf.int32, name="recipe_seq")

    embedding_layer = IngredientPositionalEmbedding(
        vocab_size=vocab_size,
        embed_dim=embedding_dim,
        max_seq_len=max_seq_len,
        name="ingredient_pos_emb",
    )
    user_embed = embedding_layer(user_input)
    recipe_embed = embedding_layer(recipe_input)

    emb_dropout = tf.keras.layers.Dropout(dropout_rate, name="emb_dropout")
    user_embed = emb_dropout(user_embed)
    recipe_embed = emb_dropout(recipe_embed)

    encoder_blocks = [
        TransformerEncoderBlock(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout_rate=dropout_rate,
            name=f"transformer_block_{index}",
        )
        for index in range(num_blocks)
    ]

    user_encoded = user_embed
    recipe_encoded = recipe_embed
    for block in encoder_blocks:
        user_encoded = block(user_encoded)
        recipe_encoded = block(recipe_encoded)

    pooling = tf.keras.layers.GlobalAveragePooling1D(name="global_avg_pool")
    user_repr = pooling(user_encoded)
    recipe_repr = pooling(recipe_encoded)

    abs_diff = AbsDiff(name="abs_diff")([user_repr, recipe_repr])
    elementwise_mul = tf.keras.layers.Multiply(name="hadamard")([user_repr, recipe_repr])

    concat = tf.keras.layers.Concatenate(name="interaction_concat")(
        [user_repr, recipe_repr, abs_diff, elementwise_mul]
    )

    x = tf.keras.layers.Dense(128, activation="relu", name="head_dense_1")(concat)
    x = tf.keras.layers.Dropout(dropout_rate, name="head_dropout")(x)
    x = tf.keras.layers.Dense(64, activation="relu", name="head_dense_2")(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="coverage_score")(x)

    return tf.keras.Model(inputs=[user_input, recipe_input], outputs=output, name="ingredient_transformer")


def binary_coverage_accuracy(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    y_true_bin = tf.cast(y_true >= 0.5, tf.float32)
    y_pred_bin = tf.cast(y_pred >= 0.5, tf.float32)
    return tf.reduce_mean(tf.cast(tf.equal(y_true_bin, y_pred_bin), tf.float32))
