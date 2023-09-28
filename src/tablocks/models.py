from typing import Iterable
import tensorflow as tf
from tensorflow.keras import layers
from . import blocks

# %%


class BaseAutoEncoder(tf.keras.Model):
    def __init__(self,
                 cats_length: Iterable[int],
                 nums_length: int,
                 cats_embedding_dim: int,
                 nums_dim: int,
                 mha_block_num_heads: int,
                 mha_num_blocks: int,
                 dropout_rate: float,
                 enc_dec_num_steps: int,
                 enc_vecs_dim: int,
                 **kwargs):

        super().__init__(**kwargs)

        self.inputs = [
            tf.keras.Input(shape=(1,), dtype=tf.float32) for _
            in cats_length] + [
                tf.keras.Input(shape=(nums_length,), dtype=tf.float32)
        ]

        self.transformer = blocks.TransformerBlock(
            cats_length,
            cats_embedding_dim,
            nums_dim,
            mha_block_num_heads,
            mha_num_blocks,
            dropout_rate)

        cats_dim = cats_length.shape[0] * cats_embedding_dim
        features_dim = cats_dim + nums_dim

        self.encoder = blocks.EncoderDecoder(
            num_steps=enc_dec_num_steps,
            min_dim=enc_vecs_dim,
            max_dim=features_dim,
            dr_rate=dropout_rate)

        self.decoder = blocks.EncoderDecoder(
            num_steps=enc_dec_num_steps,
            min_dim=enc_vecs_dim,
            max_dim=features_dim,
            dr_rate=dropout_rate,
            is_decoder=True)

        self.cats_preds = [
            layers.Dense(v, activation='softmax') for v
            in cats_length]

        self.num_preds = layers.Dense(
            nums_length,
            activation='sigmoid')

    def call(self, inputs, *args, **kwargs):

        features = self.transformer(inputs)

        encoded = self.encoder(features)

        decoded = self.decoder(encoded)

        c_preds = [lr(decoded) for lr in self.cats_preds]

        n_preds = [self.num_preds(decoded)]

        return c_preds + n_preds

# %%


class BaseExtractor(tf.keras.Model):
    def __init__(self, base_model, **kwargs):

        super().__init__(**kwargs)

        self.transformer = base_model.transformer

        self.encoder = base_model.encoder

    def call(self, inputs, *args, **kwargs):

        x = self.transformer(inputs)

        x = self.encoder(x)

        return x
