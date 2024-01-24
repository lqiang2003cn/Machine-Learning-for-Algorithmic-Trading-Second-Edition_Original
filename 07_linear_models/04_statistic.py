import warnings

import numpy as np

warnings.filterwarnings('ignore')

import pandas as pd

from statsmodels.api import OLS, add_constant, graphics
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import norm

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('whitegrid')
idx = pd.IndexSlice

with pd.HDFStore('data.h5') as store:
    data = (store['model_data']
            .dropna()
            .drop(['open', 'close', 'low', 'high'], axis=1))

data = data[data.dollar_vol_rank < 100]
data.info()

y = data.filter(like='target')
X = data.drop(y.columns, axis=1)
X = X.drop(['dollar_vol', 'dollar_vol_rank', 'volume', 'consumer_durables'], axis=1)

sns.clustermap(y.corr(), cmap=sns.diverging_palette(h_neg=20, h_pos=220), center=0, annot=True, fmt='.2%')
plt.savefig("explore.png")

sns.clustermap(X.corr(), cmap=sns.diverging_palette(h_neg=20, h_pos=220), center=0)
plt.gcf().set_size_inches((14, 14))
plt.savefig("explore2.png")

corr_mat = X.corr().stack().reset_index()
corr_mat.columns = ['var1', 'var2', 'corr']
corr_mat = corr_mat[corr_mat.var1 != corr_mat.var2].sort_values(by='corr', ascending=False)

df = pd.concat([corr_mat, corr_mat.tail()], ignore_index=True)

y.boxplot()
plt.savefig("explore3.png")

sectors = X.iloc[:, -10:]
X = (X.drop(sectors.columns, axis=1)
     .groupby(level='ticker')
     .transform(lambda x: (x - x.mean()) / x.std())
     .join(sectors)
     .fillna(0))

target = 'target_1d'
model = OLS(endog=y[target].astype(float), exog=add_constant(X).astype(float))
trained_model = model.fit()
print(trained_model.summary())

target = 'target_5d'
model = OLS(endog=y[target].astype(float), exog=add_constant(X).astype(float))
trained_model = model.fit()
print(trained_model.summary())

preds = trained_model.predict(add_constant(X))
residuals = y[target] - preds
fig, axes = plt.subplots(ncols=2, figsize=(14,4))
sns.distplot(residuals, fit=norm, ax=axes[0], axlabel='Residuals', label='Residuals')
axes[0].set_title('Residual Distribution')
axes[0].legend()
plot_acf(residuals, lags=10, zero=False, ax=axes[1], title='Residual Autocorrelation')
axes[1].set_xlabel('Lags')
sns.despine()
fig.tight_layout()

target = 'target_10d'
model = OLS(endog=y[target].astype(float), exog=add_constant(X).astype(float))
trained_model = model.fit()
print(trained_model.summary())

target = 'target_21d'
model = OLS(endog=y[target].astype(float), exog=add_constant(X).astype(float))
trained_model = model.fit()
print(trained_model.summary())