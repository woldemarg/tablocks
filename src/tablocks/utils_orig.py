import time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import normalize
import tensorflow as tf


# %%

class CustomLoggingCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        elapsed_time = time.time() - self.start_time
        print(
            f'Epoch {epoch + 1}/{self.params["epochs"]} - {elapsed_time:.0f}s {self.params["steps"]} steps - loss: {logs["loss"]:.4f} - val_loss: {logs["val_loss"]:.4f}')


# %%

def calculate_entropy(series: pd.Series) -> float:

    category_counts = series.value_counts()
    category_probabilities = category_counts / len(series)
    return stats.entropy(category_probabilities, base=2)


def calculate_weights(categorical_series, numerical_series):
    # Calculate entropy for categorical data
    categorical_entropy = (
        pd.DataFrame(categorical_series)
        .apply(calculate_entropy))

    # Calculate entropy for numerical data
    numerical_entropy = (
        pd.DataFrame(numerical_series)
        .apply(pd.qcut, q=25, duplicates='drop')
        .apply(calculate_entropy))

    # Combine and normalize the entropy values
    max_numerical_entropy = numerical_entropy.max()

    weights = normalize(
        (np.concatenate(
            [categorical_entropy.values,
             [max_numerical_entropy]])
         .reshape(-1, 1)),
        axis=0,
        norm='l1'
    ).ravel()

    return weights
