from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_openml
from sklearn.compose import ColumnTransformer
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import SVC

iris = load_iris()

X, y = iris.data, iris.target

# This dataset is way too high-dimensional. Better do PCA:
pca = PCA(n_components=1)

# Maybe some original features were good, too?
selection = SelectKBest(k=1)

# Build estimator from PCA and Univariate selection:

combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])

# Use combined features to transform dataset:
X_features = combined_features.fit(X, y).transform(X)


# %%
# Author: Pedro Morales <part.morales@gmail.com>
#
# License: BSD 3 clause


np.random.seed(0)

# Load data from https://www.openml.org/d/40945
X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)

numeric_features = ['age', 'fare']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = ['embarked', 'sex', 'pclass']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

categorical_transformer = Pipeline([
    ('imputer',
     SimpleImputer(strategy='most_frequent')
     .set_output(transform='pandas')
     ),
    ('ohe', OneHotEncoder(handle_unknown='ignore'))
])


categorical_transformer = Pipeline([
    # ('ordinal_encoder',
    #  OrdinalEncoder(
    #      handle_unknown='use_encoded_value',
    #      min_frequency=0.01,
    #      unknown_value=np.nan)
    #  .set_output(transform='pandas')
    #  ),
    ('imputer',
     SimpleImputer(strategy='most_frequent')
     .set_output(transform='pandas')
     )
])

numerical_transformer = Pipeline([
    ('imputer',
     SimpleImputer(strategy='mean')
     .set_output(transform='pandas')
     ),
    # ('power_transform',
    #  PowerTransformer()
    #  .set_output(transform='pandas')
    #  ),
    # ('scaler',
    #  MinMaxScaler()
    #  .set_output(transform='pandas')
    #  )
])
