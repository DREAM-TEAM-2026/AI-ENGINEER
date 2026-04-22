from __future__ import annotations

import tensorflow as tf


class CoverageRankingLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        mse_weight: float = 0.6,
        rank_weight: float = 0.4,
        margin: float = 0.05,
        name: str = "coverage_ranking_loss",
    ):
        super().__init__(name=name)
        self.mse_weight = mse_weight
        self.rank_weight = rank_weight
        self.margin = margin

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(tf.reshape(y_true, (-1, 1)), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, (-1, 1)), tf.float32)

        mse = tf.reduce_mean(tf.square(y_true - y_pred))

        true_diff = y_true - tf.transpose(y_true)
        pred_diff = y_pred - tf.transpose(y_pred)

        pos_mask = tf.cast(true_diff > 0.0, tf.float32)
        hinge = tf.nn.relu(self.margin - pred_diff) * pos_mask
        n_pos = tf.reduce_sum(pos_mask)
        rank_loss = tf.cond(
            n_pos > 0.0,
            lambda: tf.reduce_sum(hinge) / n_pos,
            lambda: tf.constant(0.0, dtype=tf.float32),
        )

        return self.mse_weight * mse + self.rank_weight * rank_loss

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "mse_weight": self.mse_weight,
                "rank_weight": self.rank_weight,
                "margin": self.margin,
            }
        )
        return config
