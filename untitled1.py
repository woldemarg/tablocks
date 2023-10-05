import numpy as np
import shap
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from sklearn.datasets import make_regression

# %%


def get_dataset():
    # Create sample data with sklearn make_regression function
    X, y = make_regression(
        n_samples=1000, n_features=10, n_informative=7, n_targets=5, random_state=0
    )

    # Convert the data into Pandas Dataframes for easier maniplution and keeping stored column names
    # Create feature column names
    feature_cols = [
        "feature_01",
        "feature_02",
        "feature_03",
        "feature_04",
        "feature_05",
        "feature_06",
        "feature_07",
        "feature_08",
        "feature_09",
        "feature_10",
    ]

    df_features = pd.DataFrame(data=X, columns=feature_cols)

    # Create lable column names and dataframe
    label_cols = ["labels_01", "labels_02",
                  "labels_03", "labels_04", "labels_05"]

    df_labels = pd.DataFrame(data=y, columns=label_cols)

    return df_features, df_labels

# %%


def get_model(n_inputs, n_outputs):

    model = Sequential()

    model.add(
        Dense(
            32, input_dim=n_inputs, kernel_initializer="he_uniform", activation="relu"
        )
    )
    model.add(Dense(n_outputs, kernel_initializer="he_uniform"))

    random_numbers = np.random.rand(10)
    normalized_numbers = random_numbers / np.sum(random_numbers)

    model.compile(loss="mae", optimizer="adam", loss_weights=normalized_numbers)

    return model

# %%


# Create the datasets
X, y = get_dataset()

# Get the number of inputs and outputs from the dataset
n_inputs, n_outputs = X.shape[1], y.shape[1]

# %%

model = get_model(n_inputs, n_outputs)

# %%

model.fit(X, y, verbose=1, epochs=100)

# %%

model.evaluate(x=X, y=y)

# %%

model.predict(X.iloc[0:1, :])

# %%

# print the JS visualization code to the notebook
# shap.initjs()

# %%

explainer = shap.KernelExplainer(
    model=model.predict, data=X.head(50), link="identity")

shap_values = explainer.shap_values(X=X.iloc[0:50, :], nsamples=100)

shap.summary_plot(
    shap_values=shap_values[0], features=X.iloc[0:50, :]
)

model.inputs
