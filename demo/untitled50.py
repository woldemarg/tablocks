import shap
from sklearn.datasets import fetch_openml
import math
import scipy
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    normalize,
    PowerTransformer,
    MinMaxScaler,
    OrdinalEncoder)
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# %%

# https://www.kaggle.com/c/petfinder-adoption-prediction/discussion/88740
# https://arxiv.org/abs/2012.06678


# %%

RND = 1234

cats_embedding_dim = 32

num_blcks = 2
num_heads = 4
num_steps = 4
num_epchs = 25

batch_size = 32


dropout_rate = 0.1
learning_rate = 1e-3

# %%

raw = fetch_openml(name='adult', version=4, parser='auto')
# raw = fetch_openml(name='KDDCup09_churn', parser='auto')

data, target = (raw['data'], raw['target'])

# %%

categorical_cols = data.select_dtypes(exclude=['number']).columns.tolist()

numerical_cols = data.select_dtypes(include=['number']).columns.tolist()

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

cats_transformed_test = pd.DataFrame(
    (categorical_transformers
     .transform(X_test[categorical_cols])),
    columns=categorical_cols)

nums_transformed_test = (numerical_transformer
                         .transform(X_test[numerical_cols]))

# %%

cats_length = cats_transformed.nunique().to_dict()

nums_length = nums_transformed.shape[1]

# %%

encoded_dims = 2**math.ceil(math.log2(data.shape[1]))


# %%

def create_cats_input_head(
        feature_name,
        layer,
        embedding_dim=cats_embedding_dim):

    # input_layer = keras.Input(shape=(1,), dtype=tf.float32)

    embedding_layer = layers.Embedding(
        input_dim=cats_length[feature_name],
        output_dim=embedding_dim)(layer)

    return embedding_layer


# %%

def create_mlp_block(
        hidden_units,
        normalization_layer,
        dr_rate=dropout_rate,
        activation='relu'):

    mlp_layers = [normalization_layer]

    for units in hidden_units:

        mlp_layers.append(
            layers.Dense(
                units,
                activation=activation))

    mlp_layers.append(layers.Dropout(dr_rate))

    return keras.Sequential(mlp_layers)


# %%

def create_mha_block(
    layer_out,
    n_heads=num_heads,
    num_dims=cats_embedding_dim,
    dr_rate=dropout_rate
):
    # create a multi-head attention layer
    attention_output = layers.MultiHeadAttention(
        num_heads=n_heads,
        key_dim=num_dims,
        dropout=dr_rate,
    )(layer_out, layer_out)

    # skip connection 1
    layer_odd = layers.Add()(
        [attention_output,
         layer_out])

    # layer normalization 1
    layer_odd = layers.LayerNormalization(epsilon=1e-6)(layer_odd)

    # feedforward
    feedforward_output = create_mlp_block(
        hidden_units=[num_dims],
        normalization_layer=layers.LayerNormalization(epsilon=1e-6))(layer_odd)

    # skip connection 2
    layer_odd = layers.Add()(
        [feedforward_output,
         layer_odd])

    # layer normalization 2
    layer_out = layers.LayerNormalization(epsilon=1e-6)(layer_odd)

    return layer_out


# %%

def create_model_inputs():
    inputs = {}
    for feature_name in FEATURE_NAMES:
        if feature_name in NUMERIC_FEATURE_NAMES:
            inputs[feature_name] = layers.Input(
                name=feature_name, shape=(), dtype=tf.float32
            )
        else:
            inputs[feature_name] = layers.Input(
                name=feature_name, shape=(), dtype=tf.string
            )
    return inputs


def encode_inputs(inputs, embedding_dims):

    encoded_categorical_feature_list = []
    numerical_feature_list = []

    for feature_name in inputs:
        if feature_name in CATEGORICAL_FEATURE_NAMES:

            # Get the vocabulary of the categorical feature.
            vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name]

            # Create a lookup to convert string values to an integer indices.
            # Since we are not using a mask token nor expecting any out of vocabulary
            # (oov) token, we set mask_token to None and  num_oov_indices to 0.
            lookup = layers.StringLookup(
                vocabulary=vocabulary,
                mask_token=None,
                num_oov_indices=0,
                output_mode="int",
            )

            # Convert the string input values into integer indices.
            encoded_feature = lookup(inputs[feature_name])

            # Create an embedding layer with the specified dimensions.
            embedding = layers.Embedding(
                input_dim=len(vocabulary), output_dim=embedding_dims
            )

            # Convert the index values to embedding representations.
            encoded_categorical_feature = embedding(encoded_feature)
            encoded_categorical_feature_list.append(encoded_categorical_feature)

        else:

            # Use the numerical features as-is.
            numerical_feature = tf.expand_dims(inputs[feature_name], -1)
            numerical_feature_list.append(numerical_feature)

    return encoded_categorical_feature_list, numerical_feature_list


# %%
# https://keras.io/examples/structured_data/tabtransformer/
inputs = tf.keras.Input(shape=(14,), dtype=tf.float32)

cats_input_layers = inputs[:, :8]

# cats_input_layers, cats_embeddings = zip(
#     *[create_cats_input_head(feature) for feature
#       in cats_transformed])

cats_embeddings = [create_cats_input_head(
    f, cats_input_layers[:, i]) for i, f in enumerate(cats_transformed)]

# list(enumerate(cats_transformed))

cats_features = tf.stack(cats_embeddings, axis=1)

for block in range(num_blcks):
    cats_features = create_mha_block(cats_features)

# flatten the "contextualized" embeddings of the categorical features
cats_features = layers.Flatten()(cats_features)

nums_input_layers = inputs[:, 8:]


nums_layer = [tf.expand_dims(nums_input_layers[:, i], -1)
              for i in range(nums_input_layers.shape[-1])]

nums_features = layers.concatenate(nums_layer)

# nums_features = layers.LayerNormalization(epsilon=1e-6)(nums_features)

# nums_features = layers.Dense(
#     2**math.ceil(math.log2(8 * nums_length)),
#     activation='relu')(nums_features)

nums_features = create_mlp_block(
    hidden_units=[128],
    activation='relu',
    normalization_layer=layers.LayerNormalization(epsilon=1e-6))(nums_features)

features = layers.concatenate([
    cats_features,
    nums_features
])

# Calculate the factors by which the nodes will be reduced
powers = np.unique(
    np.floor(
        np.linspace(
            math.floor(math.log2(features.shape[-1])),
            math.floor(math.log2(encoded_dims)),
            num_steps,
            endpoint=False)))[::-1]

encoded = create_mlp_block(
    hidden_units=[2**p for p in powers],
    activation='relu',
    normalization_layer=layers.BatchNormalization())(features)

vectors = layers.Dense(
    encoded_dims,
    activation='relu',
    name='encoded_vectors')(encoded)

decoded = create_mlp_block(
    hidden_units=[2**p for p in reversed(powers)],
    activation='relu',
    normalization_layer=layers.BatchNormalization())(vectors)

cats_preds = [
    layers.Dense(
        cats_length[feature],
        activation='softmax')(decoded) for feature in cats_transformed]

num_preds = [
    layers.Dense(
        1,
        activation='sigmoid')(decoded) for i in range(6)]


# %%

model = keras.Model(
    inputs=inputs,
    outputs=cats_preds + num_preds)

# %%

keras.utils.plot_model(
    model,
    show_shapes=True,
    show_dtype=False,
    show_layer_names=True,
    rankdir='LR',
    # rankdir='TD',
    # expand_nested=True,
    layer_range=None,
    show_layer_activations=False,
    # show_trainable=False
)

# %%

# model.summary()

# %%

callback = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    min_delta=0.01,
    restore_best_weights=True)

# %%

cats_loss = {
    k + '_y': 'sparse_categorical_crossentropy' for k
    in cats_transformed}


# %%

def get_entropy(series):
    category_counts = series.value_counts()
    category_probabilities = category_counts / len(series)
    return scipy.stats.entropy(category_probabilities, base=2)


cats_entropy = pd.DataFrame(cats_transformed).apply(get_entropy)

nums_entropy = (pd.DataFrame(nums_transformed)
                .apply(pd.qcut, q=25, duplicates='drop')
                .apply(get_entropy)
                .max())

entropies = normalize(
    np.array(cats_entropy.to_list() + [nums_entropy]).reshape(-1, 1),
    axis=0,
    norm='l1')

weights = pd.Series(
    entropies.ravel(),
    index=[i + '_y' for i in cats_entropy.index] + ['numericals_y']).to_dict()

# %%

optimizer = keras.optimizers.Adam(
    learning_rate=learning_rate)

model.compile(
    optimizer=optimizer,
    loss={
        **cats_loss,
        **{'numericals_y': 'mae'}},
    loss_weights=weights
)


# %%

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001),
    loss=['sparse_categorical_crossentropy'] * 8 + ['mae']*6,
    loss_weights=loss_weights)

# %%

# https://pyimagesearch.com/2019/10/14/why-is-my-validation-loss-lower-than-my-training-loss/

# https://www.kaggle.com/code/wrosinski/baselinemodeling
# https://towardsdatascience.com/predicting-adoption-speed-for-petfinder-bb4d5befb78c
# https://www.tensorflow.org/tutorials/structured_data/preprocessing_layers

history = model.fit(
    x=[s for _, s in cats_transformed.items()] + [nums_transformed],
    y={
        **{k + '_y': v for k, v in cats_transformed.items()},
        **{'numericals_y': nums_transformed}
    },
    epochs=num_epchs,
    batch_size=batch_size,
    validation_split=0.2,
    callbacks=[callback]
)

# %%

history = model.fit(
    x=train,
    y=[train[c] for c in train],
    epochs=num_epchs,
    batch_size=64,
    validation_split=0.2,
    callbacks=[callback]
)

# %%


# Create a new model with the desired layer as the output
feature_extractor_model = keras.Model(
    inputs=model.input,
    outputs=model.get_layer('encoded_vectors').output)


explainer = shap.KernelExplainer(
    model=feature_extractor_model.predict, data=train.head(100), link="identity")

shap_values = explainer.shap_values(
    X=train.iloc[0:100, :], nsamples=100, gc_collect=True)

shap.summary_plot(
    shap_values=shap_values[-4], features=train.iloc[0:100, :]
)

explainer.expected_value

# %%

data_encoded = (feature_extractor_model
                .predict(
                    [s for _, s in cats_transformed_test.items()] +
                    [nums_transformed_test]
                ))

# %%

class_weights = compute_class_weight(
    'balanced',
    classes=pd.Series(y_test).unique(),
    y=y_test)

class_weights_dct = dict(zip(
    pd.Series(y_test).unique(),
    class_weights))

# %%

clf = RandomForestClassifier(
    oob_score=True,
    n_jobs=-1,
    class_weight=class_weights_dct,
    random_state=1234)

clf.fit(data_encoded, y_test)
# clf.fit(X_test, y_test)

# https://www.kaggle.com/code/jieyima/income-classification-model
clf.oob_score_
