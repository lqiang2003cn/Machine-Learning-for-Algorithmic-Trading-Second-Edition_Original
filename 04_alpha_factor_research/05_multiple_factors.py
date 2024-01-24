from zipline.pipeline import Pipeline, CustomFactor

import numpy as np
import pandas as pd
from time import time

from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.factors import Returns, SimpleMovingAverage

# periods in trading days
MONTH = 21
QTR = 4 * MONTH
YEAR = 12 * MONTH


class AggregateFundamentals(CustomFactor):
    def compute(self, today, assets, out, inputs):
        out[:] = inputs[0]


class MeanReversion(CustomFactor):
    inputs = [Returns(window_length=MONTH)]
    window_length = YEAR

    def compute(self, today, assets, out, monthly_returns):
        df = pd.DataFrame(monthly_returns)
        out[:] = df.iloc[-1].sub(df.mean()).div(df.std())


def compute_factors():
    universe = QTradableStocksUS()

    profitability = (AggregateFundamentals(inputs=[income_statement.gross_profit],
                                           window_length=YEAR) /
                     balance_sheet.total_assets.latest).rank(mask=universe)

    roic = operation_ratios.roic.latest.rank(mask=universe)

    ebitda_yield = (AggregateFundamentals(inputs=[income_statement.ebitda],
                                          window_length=YEAR) /
                    USEquityPricing.close.latest).rank(mask=universe)

    mean_reversion = MeanReversion().rank(mask=universe)

    price_momentum = Returns(window_length=QTR).rank(mask=universe)

    sentiment = SimpleMovingAverage(inputs=[stocktwits.bull_minus_bear],
                                    window_length=5).rank(mask=universe)

    factor = profitability + roic + ebitda_yield + mean_reversion + price_momentum + sentiment

    return Pipeline(
        columns={'Profitability': profitability,
                 'ROIC': roic,
                 'EBITDA Yield': ebitda_yield,
                 "Mean Reversion (1M)": mean_reversion,
                 'Sentiment': sentiment,
                 "Price Momentum (3M)": price_momentum,
                 'Alpha Factor': factor})

start_timer = time()
start = pd.Timestamp("2015-01-01")
end = pd.Timestamp("2018-01-01")
results = compute_factors()
results.index.names = ['date', 'security']
print("Pipeline run time {:.2f} secs".format(time() - start_timer))