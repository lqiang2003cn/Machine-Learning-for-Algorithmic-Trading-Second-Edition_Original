import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from statsmodels.api import OLS, add_constant
import pandas_datareader.data as web

from linearmodels.asset_pricing import LinearFactorModel

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

ff_factor = 'F-F_Research_Data_5_Factors_2x3'
ff_factor_data = web.DataReader(ff_factor, 'famafrench', start='2010', end='2017-12')[0]
ff_factor_data.info()

ff_portfolio = '17_Industry_Portfolios'
ff_portfolio_data = web.DataReader(ff_portfolio, 'famafrench', start='2010', end='2017-12')[0]
ff_portfolio_data = ff_portfolio_data.sub(ff_factor_data.RF, axis=0)
ff_portfolio_data.info()

ff_factor_data = ff_factor_data.drop('RF', axis=1)
ff_factor_data.info()

betas = []
for industry in ff_portfolio_data:
    step1 = OLS(endog=ff_portfolio_data.loc[ff_factor_data.index, industry],
                exog=add_constant(ff_factor_data)).fit()
    betas.append(step1.params.drop('const'))

betas = pd.DataFrame(betas,
                     columns=ff_factor_data.columns,
                     index=ff_portfolio_data.columns)
betas.info()

lambdas = []
for period in ff_portfolio_data.index:
    step2 = OLS(endog=ff_portfolio_data.loc[period, betas.index],
                exog=betas).fit()
    lambdas.append(step2.params)

lambdas = pd.DataFrame(lambdas,
                       index=ff_portfolio_data.index,
                       columns=betas.columns.tolist())
lambdas.info()

lambdas.mean().sort_values().plot.barh(figsize=(12, 4))
sns.despine()
plt.tight_layout()
plt.savefig("risk.png")

t = lambdas.mean().div(lambdas.std())
print(t)

window = 24  # months
ax1 = plt.subplot2grid((1, 3), (0, 0))
ax2 = plt.subplot2grid((1, 3), (0, 1), colspan=2)
lambdas.mean().sort_values().plot.barh(ax=ax1)
lambdas.rolling(window).mean().dropna().plot(
    lw=1,
    figsize=(14, 5),
    sharey=True,
    ax=ax2
)
sns.despine()
plt.tight_layout()
plt.savefig("results.png")

window = 24  # months
lambdas.rolling(window).mean().dropna().plot(
    lw=2,
    figsize=(14, 7),
    subplots=True,
    sharey=True
)
sns.despine()
plt.tight_layout()
plt.savefig("results2.png")

mod = LinearFactorModel(
    portfolios=ff_portfolio_data,
    factors=ff_factor_data)
res = mod.fit()
print(res)
print(res.full_summary)
print(lambdas.mean())