from typing import Iterable
import math
import scipy
import pandas as pd
import numpy as np
# from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    normalize,
    PowerTransformer,
    MinMaxScaler,
    OrdinalEncoder)
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import fetch_openml


# %%

# https://www.kaggle.com/c/petfinder-adoption-prediction/discussion/88740
# https://arxiv.org/abs/2012.06678

# https://keras.io/examples/structured_data/tabtransformer/


# %%

RND = 1234

# %%

raw = fetch_openml(name='adult', version=4, parser='auto')
# raw = fetch_openml(name='KDDCup09_churn', parser='auto')

data, target = (raw['data'], raw['target'])

# %%

categorical_cols = (data
                    .select_dtypes(include=['object'])
                    .columns
                    .tolist())

numerical_cols = (data
                  .select_dtypes(include=['number'])
                  .columns
                  .tolist())

# %%

X_train, X_test, y_train, y_test = train_test_split(
    data,
    target,
    test_size=0.3,
    random_state=RND)

# %%

categorical_transformers = Pipeline([
    ('ordinal_encoder',
     (OrdinalEncoder(
         handle_unknown='use_encoded_value',
         min_frequency=0.05,
         unknown_value=np.nan)
      .set_output(transform='pandas'))
     ),
    ('imputer',
     (SimpleImputer(strategy='most_frequent')
      .set_output(transform='pandas'))
     )
])

numerical_transformer = Pipeline([
    ('imputer',
     (SimpleImputer(strategy='mean')
      .set_output(transform='pandas'))
     ),
    ('power_transform',
     (PowerTransformer()
      .set_output(transform='pandas'))
     ),
    ('scaler',
     (MinMaxScaler()
      .set_output(transform='pandas'))
     )
])

# %%

cats_transformed = (categorical_transformers
                    .fit_transform(X_train[categorical_cols]))

nums_transformed = (numerical_transformer
                    .fit_transform(X_train[numerical_cols]))

# %%

train = pd.concat([cats_transformed, nums_transformed], axis=1)

# %%

cats_length = cats_transformed.nunique()
nums_length = nums_transformed.shape[1]


# %%

class CatInputBlock(layers.Layer):
    def __init__(self,
                 cats_length: Iterable[int],
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

        embedding_outputs = [
            embedding_layer(input_layer)
            for input_layer, embedding_layer
            in zip(
                inputs,
                self.embedding_layers)]

        return tf.stack(embedding_outputs, axis=1)


class NumericalInputBlock(layers.Layer):
    def __init__(self,
                 units: int,
                 **kwargs):

        super().__init__(**kwargs)

        self.dense_layer = layers.Dense(
            units,
            activation='relu')

    def call(self, inputs, *args, **kwargs):

        return self.dense_layer(inputs)


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


class TransformerBlock(layers.Layer):
    def __init__(self,
                 cats_length: Iterable[int],
                 cats_embedding_dim: int,
                 nums_dim: int,
                 mha_block_num_heads: int,
                 mha_num_blocks: int,
                 dropout_rate: float,
                 **kwargs):

        super().__init__(**kwargs)

        self.cats_length = cats_length
        self.nums_dim = nums_dim
        self.cats_embedding_dim = cats_embedding_dim
        self.mha_block_num_heads = mha_block_num_heads
        self.dropout_rate = dropout_rate
        self.mha_num_blocks = mha_num_blocks
        self.cats_input_block = None
        self.mha_blocks = None
        self.nums_input_block = None

    def build(self, input_shape):

        self.cats_input_block = CatInputBlock(
            self.cats_length,
            self.cats_embedding_dim)

        self.mha_blocks = [MHABlock(
            num_heads=self.mha_block_num_heads,
            units=self.cats_embedding_dim,
            dr_rate=self.dropout_rate) for _ in range(self.mha_num_blocks)]

        self.nums_input_block = NumericalInputBlock(self.nums_dim)

    def call(self, inputs, *args, **kwargs):

        cats_inputs, nums_inputs = inputs[:-1], inputs[-1]

        cats_features = self.cats_input_block(cats_inputs)

        for block in self.mha_blocks:
            cats_features = block(cats_features)

        cats_features = layers.Flatten()(cats_features)

        nums_features = self.nums_input_block(nums_inputs)

        features = layers.concatenate(
            [cats_features,
             nums_features])

        return features


# %%
enc_vecs_dim = 2**math.ceil(math.log2(data.shape[1]))
numerics_dim = 2**math.ceil(math.log2(8 * nums_length))

params = {
    'cats_embedding_dim': 32,
    'nums_dim': numerics_dim,
    'mha_block_num_heads': 4,
    'mha_num_blocks': 2,
    'dropout_rate': 0.1,
    'enc_dec_num_steps': 4,
    'enc_vecs_dim': enc_vecs_dim
}


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
            keras.Input(shape=(1,), dtype=tf.float32) for _
            in cats_length] + [
                keras.Input(shape=(nums_length,), dtype=tf.float32)
        ]

        self.transformer = TransformerBlock(
            cats_length,
            cats_embedding_dim,
            nums_dim,
            mha_block_num_heads,
            mha_num_blocks,
            dropout_rate)

        cats_dim = cats_length.shape[0] * cats_embedding_dim
        features_dim = cats_dim + nums_dim

        self.encoder = EncoderDecoder(
            num_steps=enc_dec_num_steps,
            min_dim=enc_vecs_dim,
            max_dim=features_dim,
            dr_rate=dropout_rate)

        self.decoder = EncoderDecoder(
            num_steps=enc_dec_num_steps,
            min_dim=enc_vecs_dim,
            max_dim=features_dim,
            dr_rate=dropout_rate,
            is_decoder=True)

        self.cats_preds = [
            layers.Dense(
                cats_length[feature],
                activation='softmax') for feature in cats_transformed]

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


class BaseExtractor(tf.keras.Model):
    def __init__(self, base_model, **kwargs):

        super().__init__(**kwargs)

        self.transformer = base_model.transformer
        self.encoder = base_model.encoder

    def call(self, inputs, *args, **kwargs):

        x = self.transformer(inputs)
        x = self.encoder(x)

        return x

# %%


model = BaseAutoEncoder(
    cats_length=cats_length,
    nums_length=nums_length,
    **params)

# %%

base_learning_rate = 1e-3
num_epchs = 25
batch_size = 32


# %%

def get_entropy(series):
    category_counts = series.value_counts()
    category_probabilities = category_counts / len(series)
    return scipy.stats.entropy(category_probabilities, base=2)


# %%

cats_entropy = pd.DataFrame(cats_transformed).apply(get_entropy)

nums_entropy = (pd.DataFrame(nums_transformed)
                .apply(pd.qcut, q=25, duplicates='drop')
                .apply(get_entropy)
                .max())

weights = normalize(
    np.array(cats_entropy.to_list() + [nums_entropy]).reshape(-1, 1),
    axis=0,
    norm='l1').ravel()

# %%

base_optimizer = keras.optimizers.Adam(
    learning_rate=base_learning_rate)

model.compile(
    optimizer=base_optimizer,
    loss=['sparse_categorical_crossentropy'] * cats_length.shape[0] + ['mae'],
    loss_weights=weights)

# %%

callback = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    min_delta=0.01,
    restore_best_weights=True)

# %%

# https://pyimagesearch.com/2019/10/14/why-is-my-validation-loss-lower-than-my-training-loss/

# https://www.kaggle.com/code/wrosinski/baselinemodeling
# https://towardsdatascience.com/predicting-adoption-speed-for-petfinder-bb4d5befb78c
# https://www.tensorflow.org/tutorials/structured_data/preprocessing_layers

train_set = [s for _, s in cats_transformed.items()] + [nums_transformed]

history = model.fit(
    x=train_set,
    y=train_set,
    epochs=num_epchs,
    batch_size=batch_size,
    validation_split=0.2,
    callbacks=[callback])

# %%

# model.summary()


# %%

extractor = BaseExtractor(model)
extractor.trainable = False

# %%

cats_transformed_test = pd.DataFrame(
    (categorical_transformers
     .transform(X_test[categorical_cols])),
    columns=categorical_cols)

nums_transformed_test = (numerical_transformer
                         .transform(X_test[numerical_cols]))

test_set = [s for _, s in cats_transformed_test.items()] + \
    [nums_transformed_test]

# %%

# test_encoded = extractor.predict(test_set)
data_encoded = extractor.predict(train_set)

# %%

# class_weights = compute_class_weight(
#     'balanced',
#     classes=pd.Series(y_test).unique(),
#     y=y_test)

# class_weights_dct = dict(zip(
#     pd.Series(y_test).unique(),
#     class_weights))

# %%

# clf = RandomForestClassifier(
#     oob_score=True,
#     n_jobs=-1,
#     class_weight=class_weights_dct,
#     random_state=1234)

# clf.fit(test_encoded, y_test)
# # clf.fit(X_test, y_test)

# # https://www.kaggle.com/code/jieyima/income-classification-model
# print(clf.oob_score_)

# %%

# keras.utils.plot_model(
#     clf_model,
#     show_shapes=True,
#     show_dtype=False,
#     show_layer_names=True,
#     rankdir='TD',
#     expand_nested=True,
#     layer_range=None,
#     show_layer_activations=False,
#     # show_trainable=False
# )

# %%

n_clusters = 8

kmeans = KMeans(
    n_clusters=n_clusters,
    random_state=RND,
    n_init='auto')

kmeans.fit(normalize(data_encoded))

target = kmeans.labels_

# %%

inputs = model.inputs

x = extractor(inputs, training=False)

x = MLPBlock(
    units_lst=[x.shape[1] * 2, x.shape[1]],
    normalization_layer=layers.BatchNormalization(),
    dr_rate=0.1,
    name='encoder')(x)

# x = EncoderDecoder(
#     num_steps=2,
#     min_dim=16,
#     max_dim=x.shape[1],
#     dr_rate=0.1,
#     name='encoder')(x)

outputs = layers.Dense(n_clusters, activation='softmax')(x)

clf_model = tf.keras.Model(inputs, outputs)

clf_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=keras.optimizers.Adam(
        learning_rate=base_learning_rate),
    metrics=['accuracy'])

# %%

# clf_model.summary()

# %%

class_weights = compute_class_weight(
    'balanced',
    classes=pd.Series(target).unique(),
    y=target)

class_weights_dct = dict(zip(
    pd.Series(target).unique(),
    class_weights))

# %%

h = clf_model.fit(
    x=train_set,
    y=target,
    epochs=num_epchs,
    validation_split=0.2,
    class_weight=class_weights_dct,
    callbacks=[callback]
)

# %%

extractor.trainable = True

# %%

clf_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=keras.optimizers.Adam(
        learning_rate=base_learning_rate / 10),
    metrics=['accuracy'])

# %%

h = clf_model.fit(
    x=train_set,
    y=target,
    epochs=num_epchs,
    validation_split=0.2,
    class_weight=class_weights_dct,
    callbacks=[callback]
)

# %%

# clf_model.summary()

# %%

feature_extractor_model = keras.Model(
    inputs=clf_model.input,
    outputs=clf_model.get_layer('encoder').output)

# %%

vectors = feature_extractor_model.predict(train_set)
