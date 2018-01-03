# Title: THFC (Terrible Hedge Fund Calculator)
# Author = M.A. Hervieux
# License = GNU v2
#
# QSTK Imports  -- Get rid of these later on if possible
import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkutil.DataAccess as da

# Third Party Imports
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#Increasing Precision to give result closer to numpy
pd.set_option('precision', 8)

def init_data(dt_start, dt_end, ls_symbols):
    # Setting attributes for trading days lenght and distribution
    dt_timeofday = dt.timedelta(hours=16)
    ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt_timeofday)

    # Accessing Yahoo data
    c_dataobj = da.DataAccess('Yahoo')
    ls_keys = ['close']
    ldf_data = c_dataobj.get_data(ldt_timestamps, ls_symbols, ls_keys)
    d_data = dict(zip(ls_keys, ldf_data))

    # Loading data into Dataframe
    df = pd.DataFrame(data=d_data['close'].values, index=ldt_timestamps, columns=ls_symbols)
    df.fillna(method='ffill')

    return df

def building_portfolio(dt_start, dt_end, df_price, lf_allocations):
    # Normalizing prices
    df_portfolio = df_price/ df_price.values[0, :]
    # Price in the allocation
    df_portfolio = df_portfolio * lf_allocations
    # Adding Daily Cumulative Column
    df_portfolio['DailyCum'] = df_portfolio.cumsum(axis=1).iloc[:,-1]
    # Extracting values with deep copy to preserve DailyCum
    na_rets = df_portfolio['DailyCum'].values.copy()
    # Feed Value to returnize0. The change is inplace
    tsu.returnize0(na_rets)
    # Store result in new column
    df_portfolio["DailyRets"] = na_rets
    # compute std deviation of daily returns
    df_portfolio.daily_std = df_portfolio.DailyRets.std(ddof=0)
    #compute average of daily returns
    df_portfolio.avg_daily_return = df_portfolio.DailyRets.mean()
    #get cumulative return from the last element of daily returns
    df_portfolio.cumrets = df_portfolio["DailyCum"][-1]
    #get num of trading days
    trading_days = df_price.shape[0]
    #calc the Sharpe Ratio
    df_portfolio.sharpeRatio = (np.sqrt(trading_days) * df_portfolio.avg_daily_return) / df_portfolio.daily_std

    return df_portfolio

def benchmark_cmp(dt_start, dt_end, df_portfolio, bench_index):
    df_bench_price = init_data(dt_start, dt_end, bench_index)
    df_bench = building_portfolio(dt_start, dt_end, df_bench_price,[1.0])

    # Graphing cum rets
    plt.clf()
    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(df_bench['DailyCum'].values, alpha=0.4)
    plt.plot(df_portfolio['DailyCum'].values)
    ls_names = [bench_index[0], 'Portfolio']
    plt.legend(ls_names)
    plt.ylabel('Cumulative Returns')
    plt.xlabel('Trading Day')
    fig.autofmt_xdate(rotation=45)
    plt.savefig('portfolioVSbench2.pdf', format='pdf')

def simulate(dt_start, dt_end, df_price):
#Must implement gradient descent at a later point
    MaxSharpeRatio = -1
    for x in range(0, 11):
        for y in range(0, 11 - x):
            for z in range(0, 11 - x - y):
                for a in range(0, 11 - x - y - z):
                    if (x+y+z+a) == 10:
                        alloc = [float(x) / 10, float(y) / 10, float(z) / 10, float(a) / 10]
                        df_test_portfolio = building_portfolio(dt_start, dt_end, df_price, alloc)
                        if df_test_portfolio.sharpeRatio > MaxSharpeRatio:
                            MaxSharpeRatio = df_test_portfolio.sharpeRatio
                            df_portfolio = df_test_portfolio
                            BestAlloc = alloc

        return df_portfolio, BestAlloc

def main():
    # List of symbols
    ls_symbols = ['AXP', 'HPQ', 'IBM', 'HNZ']
    bench_index = ['SPY']

    # Start and End date of the charts
    dt_start = dt.datetime(2010, 1, 1)
    dt_end = dt.datetime(2010, 12, 31)

    # Retrieiving financial data exactly once for period and symbols
    df_price = init_data(dt_start, dt_end, ls_symbols)

#    test_allocation = [.25,.25,.25,.25]

    df_portfolio, BestAlloc = simulate(dt_start, dt_end, df_price)

    print "Start Date: {0:%B %d, %Y}".format(dt_start)
    print "End Date: {0:%B %d, %Y}".format(dt_end)
    print "Symbols: {0}".format(ls_symbols)
    print "Optimal Allocations: {0}".format(BestAlloc)
    print "Sharpe Ratio: {0}".format(df_portfolio.sharpeRatio)
    print "Volatility (stdev of daily returns): {0}".format(df_portfolio.daily_std)
    print "Average Daily Return: {0}".format(df_portfolio.avg_daily_return)
    print "Cumulative Return: {0}".format(df_portfolio.cumrets)

    benchmark_cmp(dt_start, dt_end, df_portfolio, bench_index)

if __name__ == '__main__':
    main()
