import random
# import warnings
import math
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    normalize,
    PowerTransformer,
    MinMaxScaler,
    OrdinalEncoder)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from fastshap import KernelExplainer
# import shap
from prosphera.projector import Projector
from tensorflow.keras import layers
from src.tablocks import blocks, models, utils
# import ext.cluster_utils as cluster

# %%

# https://www.kaggle.com/c/petfinder-adoption-prediction/discussion/88740
# https://arxiv.org/abs/2012.06678

# https://keras.io/examples/structured_data/tabtransformer/

# https://pyimagesearch.com/2019/10/14/why-is-my-validation-loss-lower-than-my-training-loss/

# https://www.kaggle.com/code/wrosinski/baselinemodeling
# https://towardsdatascience.com/predicting-adoption-speed-for-petfinder-bb4d5befb78c
# https://www.tensorflow.org/tutorials/structured_data/preprocessing_layers


# %%

RND = 1234
tf.random.set_seed(RND)
np.random.seed(RND)
random.seed(RND)

# %%

# raw = fetch_openml(name='adult', version=4, parser='auto')
# raw = fetch_openml(name='KDDCup09_churn', parser='auto')
raw = fetch_openml(name='kick', version=1, parser='auto')
# raw = fetch_openml(name='higgs', version=1, parser='auto')

data, target = (raw['data'], raw['target'])
# data['class'] = target.astype(str)

# %%

# categorical_cols = (data
#                     .select_dtypes(exclude=['number'])
#                     .columns
#                     .tolist())

# numerical_cols = (data
#                   .select_dtypes(include=['number'])
#                   .columns
#                   .tolist())

# %%

X_train, X_test, y_train, y_test = train_test_split(
    data,
    target,
    test_size=0.3,
    random_state=RND)

# %%

categorical_transformer = Pipeline([
    ('ordinal_encoder',
     OrdinalEncoder(
         handle_unknown='use_encoded_value',
         min_frequency=0.01,
         unknown_value=np.nan)),
    ('imputer',
     SimpleImputer(strategy='most_frequent',
                   keep_empty_features=True))
])

numerical_transformer = Pipeline([
    ('imputer',
     SimpleImputer(strategy='mean',
                   keep_empty_features=True)),
    ('power_transform',
     PowerTransformer()),
    ('scaler',
     MinMaxScaler())
])

# %%

preprocessor = (ColumnTransformer(
    transformers=[
        ('cat',
         categorical_transformer,
         # categorical_cols
         make_column_selector(dtype_exclude=np.number)
         ),
        ('num',
         numerical_transformer,
         # numerical_cols
         make_column_selector(dtype_include=np.number)
         )],
    n_jobs=-1,
    verbose_feature_names_out=False)
    .set_output(transform='pandas'))


# %%

try:
    train = preprocessor.fit_transform(X_train)
except Exception as err:
    print(f'{err}')
    preprocessor.set_params(num__power_transform='passthrough')
    train = preprocessor.fit_transform(X_train)

# %%

oe = preprocessor.named_transformers_['cat'].named_steps['ordinal_encoder']
cats_length = [len(x) for x in oe.categories_]

ms = preprocessor.named_transformers_['num'].named_steps['scaler']
nums_length = ms.n_features_in_

# cats_length = train[categorical_cols].nunique()
# nums_length = len(numerical_cols)

# %%

base_units = 32

enc_vecs_dim = min(2**math.floor(math.log2(data.shape[1])), 64)
# numerics_dim = min(2**math.ceil(math.log2(base_units * nums_length)), 256)

# %%

params = {
    'cats_embedding_dim': base_units,
    # 'nums_dim': numerics_dim,
    'mha_block_num_heads': 4,
    'mha_num_blocks': 2,
    'dropout_rate': 0.1,
    'enc_dec_num_steps': 3,
    'enc_vecs_dim': enc_vecs_dim
}

# %%

autoencoder = models.BaseAutoEncoder(
    cats_length=cats_length,
    nums_length=nums_length,
    **params)

# %%

base_learning_rate = 1e-3
num_epchs = 25
batch_size = 128

# %%

loss_weights = utils.calculate_weights(train, nums_length)

# %%

autoencoder.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=base_learning_rate),
    loss=(['sparse_categorical_crossentropy'] * len(cats_length) +
          ['mae'] * nums_length),
    loss_weights=loss_weights)

# %%

loss_callback = utils.CustomLoggingCallback()

stop_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    min_delta=0.01,
    restore_best_weights=True)

# %%

# train_set = [s for _, s in cats_transformed.items()] + [nums_transformed]

# %%


# class TabularSequence(tf.keras.utils.Sequence):

#     def __init__(self, x_set, y_set, batch_size):
#         self.x, self.y = x_set, y_set
#         self.batch_size = batch_size

#     def __len__(self):
#         return math.ceil(len(self.x[0]) / self.batch_size)

#     def __getitem__(self, idx):
#         low = idx * self.batch_size
#         high = min(low + self.batch_size, len(self.x[0]))
#         batch_x = [np.array(e[low:high]).reshape(-1, 1)
#                    for e in self.x[:-1]] + [self.x[-1][low:high]]
#         batch_y = [np.array(e[low:high]).reshape(-1, 1)
#                    for e in self.y[:-1]] + [self.y[-1][low:high]]
#         return batch_x, batch_y


# %%
# self.x = test_set
# low, high = 0, 10

# len(train_set[0])

# batch_x = [np.array(e[low:high]).reshape(-1, 1)
#            for e in test_set[:-1]] + [test_set[-1][low:high]]

# %%

# tabular_data_generator = TabularSequence(train_set, train_set, batch_size)
# tabular_data_generator_val = TabularSequence(test_set, test_set, batch_size)

# %%

# d = iter(tabular_data_generator)
# dd = next(d)

# %%

# d = [np.array([1.]),
#      np.array([3.]),
#      np.array([1.]),
#      np.array([2.]),
#      np.array([0.]),
#      np.array([1.]),
#      np.array([1.]),
#      np.array([0.]),
#      np.array([[0.26912655,
#                 0.26513581,
#                 0.52726954,
#                 0.,
#                 0.,
#                 0.40411106]])]

# %%
# len(tabular_data_generator)

# autoencoder.predict(dd[0])

# %%

# aue_history = autoencoder.fit(
#     # x=train_set,
#     # y=train_set,
#     tabular_data_generator,
#     epochs=5,
#     # batch_size=batch_size,
#     # validation_split=0.2,
#     validation_data=tabular_data_generator_val,
#     # verbose=0,
#     # workers=8,
#     # use_multiprocessing=True,
#     callbacks=[loss_callback, stop_callback]
# )

aue_history = autoencoder.fit(
    x=train,
    y=[train[c] for c in train],
    # tabular_data_generator,
    epochs=num_epchs,
    batch_size=batch_size,
    validation_split=0.2,
    # validation_data=tabular_data_generator_val,
    verbose=0,
    # workers=8,
    # use_multiprocessing=True,
    callbacks=[loss_callback, stop_callback]
)

# %%

# len(tabular_data_generator_val)
# autoencoder.summary()

# %%

extractor = models.BaseExtractor(autoencoder)
extractor.trainable = False

# extractor.predict(tabular_data_generator_val)

# %%

train_encoded = extractor.predict(train)
train_encoded_norm = normalize(train_encoded)

# %%

# optimal_clusters = cluster.find_optimal_clusters(train_encoded_norm)
optimal_clusters = 7

# %%

kmeans = KMeans(
    n_clusters=optimal_clusters,
    random_state=RND,
    n_init='auto')

kmeans.fit(train_encoded_norm)

labels = kmeans.labels_

# %%

inputs = autoencoder.inputs

x = extractor(inputs, training=False)

x = blocks.MLPBlock(
    units_lst=[x.shape[1] * 2, x.shape[1]],
    normalization_layer=layers.BatchNormalization(),
    dr_rate=0.1,
    name='encoder')(x)

outputs = layers.Dense(optimal_clusters, activation='softmax')(x)

classifier = tf.keras.Model(inputs, outputs)

classifier.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=base_learning_rate),
    metrics=['accuracy'])

# %%

classifier.summary()

# %%

class_weights = compute_class_weight(
    'balanced',
    classes=pd.Series(labels).unique(),
    y=labels)

class_weights_dct = dict(zip(
    pd.Series(labels).unique(),
    class_weights))

# %%

clf_history = classifier.fit(
    x=train,
    y=labels,
    epochs=num_epchs,
    batch_size=batch_size,
    validation_split=0.2,
    class_weight=class_weights_dct,
    callbacks=[stop_callback]
)

# %%

extractor.trainable = True

classifier.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=base_learning_rate / 10),
    metrics=['accuracy'])

# %%

classifier.summary()

# %%

clf_history_tune = classifier.fit(
    x=train,
    y=labels,
    epochs=num_epchs,
    batch_size=batch_size,
    initial_epoch=clf_history.epoch[-1],
    validation_split=0.2,
    class_weight=class_weights_dct,
    callbacks=[stop_callback]
)

# %%

extractor_tuned = tf.keras.Model(
    inputs=classifier.input,
    outputs=classifier.get_layer('encoder').output)

# %%

test = preprocessor.transform(X_test)

# %%

test_cls = np.argmax(classifier.predict(test), axis=1)
test_vec = extractor_tuned.predict(test)

# %%

visualizer = Projector()

# %%

visualizer.project(
    data=test_vec,
    # data=train_encoded,
    labels=test_cls
    # labels=y_train
)


# %%

# explainer = shap.KernelExplainer(
#     model=extractor.predict,
#     data=shap.kmeans(train, 100),
#     link='identity',
#     seed=RND)


# with warnings.catch_warnings():
#     warnings.simplefilter('ignore')

#     shap_values = explainer.shap_values(
#         X=shap.sample(train, 150, random_state=RND),
#         nsamples=100,
#         silent=True,
#         gc_collect=True)

# shap.summary_plot(
#     shap_values=shap_values[2], features=train.iloc[:150, :]
# )

# %%


# d = [np.abs(np.mean(x, axis=0)).reshape(1, -1) for x
#      in shap_values if np.sum(x) != 0]

# dd = np.concatenate(d, axis=0)
# m = np.mean(dd, axis=0)
# mm = m / np.sum(m)


# pd.Series(mm, index=train.columns).sort_values().plot.barh()

# %%
ke = KernelExplainer(extractor.predict, train.sample(1000))
# sv = ke.calculate_shap_values(train.iloc[:10], verbose=False)
ke.stratify_background_set(50)
sv = ke.calculate_shap_values(
    train.iloc[:100],
    background_fold_to_use=0,
    verbose=False
)

a = sv[:, :-1, :]

s = [np.mean(normalize(np.abs(layer), norm='l1', axis=0),
             axis=1).reshape(-1, 1) for layer in a]

# Concatenate the results along axis 1
x = np.concatenate(s, axis=1)

# Calculate the mean of x and normalize it
m = np.mean(x, axis=1)
mm = m / np.sum(m)

pd.Series(mm, index=train.columns).sort_values().plot.barh()
