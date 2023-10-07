import math
from typing import Iterable
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers


# %%

class CatInputBlock(layers.Layer):
    def __init__(self,
                 cats_length: pd.Series,
                 cats_embedding_dim: int,
                 **kwargs):

        super().__init__(**kwargs)

        self.embedding_layers = [
            layers.Embedding(
                input_dim=v,
                output_dim=cats_embedding_dim)
            for v in cats_length]

    def call(self,
             inputs,
             *args,
             **kwargs):

        embedding_outputs = [lr(inputs[:, i]) for i, lr
                             in enumerate(self.embedding_layers)]

        return tf.stack(embedding_outputs, axis=1)


# %%

# class NumericalInputBlock(layers.Layer):
#     def __init__(self,
#                  units: int,
#                  num_length: int,
#                  **kwargs):

#         super().__init__(**kwargs)

#         self.layer = tf.expandlayers.Dense(
#             units,
#             activation='relu')

#     def call(self, inputs, *args, **kwargs):

#         return self.dense_layer(inputs)


# %%

class MLPBlock(layers.Layer):
    def __init__(self,
                 units_lst: Iterable[int],
                 normalization_layer: layers.Layer,
                 dr_rate: float,
                 activation: str = 'relu',
                 **kwargs):

        super().__init__(**kwargs)

        self.normalization_layer = normalization_layer

        self.dropout_layer = layers.Dropout(dr_rate)

        self.dense_layers = [
            layers.Dense(
                units,
                activation=activation) for units
            in units_lst]

    def call(self, inputs, *args, **kwargs):

        x = self.normalization_layer(inputs)

        for layer in self.dense_layers:
            x = layer(x)

        x = self.dropout_layer(x)

        return x


# %%

class MHABlock(layers.Layer):
    def __init__(self,
                 num_heads: int,
                 units: int,
                 dr_rate: float,
                 **kwargs):

        super().__init__(**kwargs)

        self.multihead_attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=units,
            dropout=dr_rate)

        self.layer_norm1 = layers.LayerNormalization(epsilon=1e-6)

        self.layer_norm2 = layers.LayerNormalization(epsilon=1e-6)

        self.mlp_block = MLPBlock(
            units_lst=[units],
            normalization_layer=layers.LayerNormalization(epsilon=1e-6),
            dr_rate=dr_rate)

    def call(self, inputs, *args, **kwargs):

        attention_output = self.multihead_attention(inputs, inputs)

        skip_connection1 = layers.Add()(
            [attention_output,
             inputs])

        layer_norm1 = self.layer_norm1(skip_connection1)

        feedforward_output = self.mlp_block(layer_norm1)

        skip_connection2 = layers.Add()(
            [feedforward_output,
             layer_norm1])

        layer_norm2 = self.layer_norm2(skip_connection2)

        return layer_norm2


# %%

class EncoderDecoder(layers.Layer):
    def __init__(self,
                 num_steps: int,
                 min_dim: int,
                 max_dim: int,
                 dr_rate: float,
                 is_decoder: bool = False,
                 activation: str = 'relu',
                 **kwargs):

        super().__init__(**kwargs)

        self.num_steps = num_steps
        self.min_dim = min_dim
        self.max_dim = max_dim
        self.dr_rate = dr_rate
        self.is_decoder = is_decoder
        self.activation = activation
        self.mlp_block = None
        self.dense_layer = None

    def build(self, input_shape):

        powers = np.unique(
            np.floor(
                np.linspace(
                    math.floor(math.log2(self.max_dim)),
                    math.floor(math.log2(self.min_dim)),
                    self.num_steps,
                    endpoint=False)
            )
        )[::-1]

        if self.is_decoder:
            powers = powers[::-1]

        self.mlp_block = MLPBlock(
            units_lst=[2**p for p in powers],
            normalization_layer=layers.BatchNormalization(),
            dr_rate=self.dr_rate)

        if not self.is_decoder:
            self.dense_layer = layers.Dense(
                self.min_dim, activation=self.activation)

    def call(self, inputs, *args, **kwargs):

        x = self.mlp_block(inputs)

        if not self.is_decoder:
            x = self.dense_layer(x)

        return x


# %%

class TransformerBlock(layers.Layer):
    def __init__(self,
                 cats_length: Iterable[int],
                 cats_embedding_dim: int,
                 nums_length: int,
                 # nums_dim: int,
                 mha_block_num_heads: int,
                 mha_num_blocks: int,
                 dropout_rate: float,
                 **kwargs):

        super().__init__(**kwargs)

        self.cats_length = cats_length
        self.cats_embedding_dim = cats_embedding_dim
        self.nums_length = nums_length
        # self.nums_dim = nums_dim
        self.mha_block_num_heads = mha_block_num_heads
        self.dropout_rate = dropout_rate
        self.mha_num_blocks = mha_num_blocks
        self.cats_input_block = None
        self.mha_blocks = None

    def build(self, input_shape):

        self.cats_input_block = CatInputBlock(
            self.cats_length,
            self.cats_embedding_dim)

        self.mha_blocks = [MHABlock(
            num_heads=self.mha_block_num_heads,
            units=self.cats_embedding_dim,
            dr_rate=self.dropout_rate) for _ in range(self.mha_num_blocks)]

        # self.mlp_block = MLPBlock(
        #     units_lst=[self.nums_dim],
        #     normalization_layer=layers.LayerNormalization(epsilon=1e-6),
        #     dr_rate=self.dropout_rate)

        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, *args, **kwargs):

        cats_inputs, nums_inputs = (inputs[:, :len(self.cats_length)],
                                    inputs[:, len(self.cats_length):])

        cats_features = self.cats_input_block(cats_inputs)

        for block in self.mha_blocks:
            cats_features = block(cats_features)

        cats_features = layers.Flatten()(cats_features)

        # cats_features = layers.Dense(
        #     self.nums_dim,
        #     activation='relu')(cats_features)

        nums_features = [tf.expand_dims(nums_inputs[:, i], -1) for i
                         in range(self.nums_length)]

        nums_features = layers.concatenate(nums_features)

        nums_features = self.layer_norm(nums_features)

        # nums_features = self.mlp_block(nums_features)

        features = layers.concatenate(
            [cats_features,
             nums_features])

        # weights = layers.Dense(cats_features.shape[-1])(features)

        # feat1ures = tf.multiply(features, weights)

        return features
