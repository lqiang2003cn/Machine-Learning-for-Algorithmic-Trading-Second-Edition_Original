import warnings

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import mutual_info_classif

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
idx = pd.IndexSlice

with pd.HDFStore('../data/assets.h5') as store:
    data = store['engineered_features']

dummy_data = pd.get_dummies(
    data,
    columns=['year', 'month', 'msize', 'age', 'sector'],
    prefix=['year', 'month', 'msize', 'age', ''],
    prefix_sep=['_', '_', '_', '_', '']
)
dummy_data = dummy_data.rename(columns={c: c.replace('.0', '') for c in dummy_data.columns})
dummy_data.info()

target_labels = [f'target_{i}m' for i in [1, 2, 3, 6, 12]]
targets = data.dropna().loc[:, target_labels]

features = data.dropna().drop(target_labels, axis=1)
features.sector = pd.factorize(features.sector)[0]

cat_cols = ['year', 'month', 'msize', 'age', 'sector']
discrete_features = [features.columns.get_loc(c) for c in cat_cols]

mutual_info = pd.DataFrame()
for label in target_labels:
    mi = mutual_info_classif(
        X=features,
        y=(targets[label] > 0).astype(int),
        discrete_features=discrete_features,
        random_state=42
    )
    mutual_info[label] = pd.Series(mi, index=features.columns)

mutual_info.sum()

fig, ax = plt.subplots(figsize=(15, 4))
sns.heatmap(mutual_info.div(mutual_info.sum()).T, ax=ax, cmap='Blues')
plt.savefig("heatmap.png")
