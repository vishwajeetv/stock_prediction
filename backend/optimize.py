import Quandl
import numpy as np
import array
import matplotlib.pyplot as plotter
import pandas
import csv
import math
import scipy.optimize as optimizer

def readData(file, startDate, endDate):
    prices = pandas.read_csv(file, parse_dates=True, index_col='Date', usecols=['Date', 'Close Price']).fillna(method='ffill')
    prices = prices.ix[endDate:startDate]
    print(prices)
    prices = normalize(prices)
    return prices

def normalize(df):
    dfToDivide = df['Close Price']
    normalized = dfToDivide.divide(dfToDivide[-1])
    return normalized

def optimizeCumulativeReturns(allocs, allStocks):
    allocs = allStocks * allocs
    postvalues = allocs * 100000
    portfolioValues = postvalues.sum(axis=1)
    cumulativeReturns = (portfolioValues[-1] / portfolioValues[0]) - 1
    return cumulativeReturns*-1

def optimizeSharpe(allocs, allStocks):
    allocs = allStocks * allocs
    postvalues = allocs * 100000
    portfolioValues = postvalues.sum(axis=1)

    dailyReturns = (portfolioValues / portfolioValues.shift(1)) - 1
    dailyReturns = dailyReturns.ix[0:]

    dailyReturnsStD = dailyReturns.std()
    dailyReturnsMean = dailyReturns.mean()
    sharpeRatio = dailyReturnsMean / dailyReturnsStD
    return sharpeRatio*-1

def optimizePortfolio(allocs, startvalue, allStocks):

    allocsGuess = allocs

    cons = ({'type': 'eq', 'fun' : lambda inputs: 1 - (np.sum(np.absolute(inputs)))})
    bnds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]

    #optimize for cumulative returns
    optimizedOutput = optimizer.minimize(optimizeCumulativeReturns, allocsGuess, args=(allStocks,) ,
                                       method='SLSQP',  bounds=bnds, constraints=cons,
                                        options={'disp': True})

    #optimize for sharpe ratio
    # optimizedOutput = optimizer.minimize(optimizeSharpe, allocsGuess, args=(allStocks,) ,
    #                                        method='SLSQP',  bounds=bnds, constraints=cons,
    #                                         options={'disp': True})

    # print("X = {}, Y = {}".format(optimizedOutput.x,optimizedOutput.fun))
    print("REL :"+'{:f}'.format(optimizedOutput.x[0]))
    print("HDFC :"+'{:f}'.format(optimizedOutput.x[1]))
    print("SUN :"+'{:f}'.format(optimizedOutput.x[2]))
    print("ONGC :"+'{:f}'.format(optimizedOutput.x[3]))

    optimizedAllocations = optimizedOutput.x
    allocationsApplied = allStocks * optimizedAllocations
    postvalues = allocationsApplied * startvalue
    portfolioValues = postvalues.sum(axis=1)
    print(portfolioValues.head())
    cumulativeReturns = (portfolioValues[-1] / portfolioValues[0]) - 1
    print(cumulativeReturns)
    plotter.plot(portfolioValues)
    plotter.show()
    dailyReturns = (portfolioValues / portfolioValues.shift(1)) - 1
    dailyReturns = dailyReturns[1:]

    dailyReturnsStD = dailyReturns.std()
    dailyReturnsMean = dailyReturns.mean()
    #252 trading days
    k = math.sqrt(250)
    sharpeRatio = dailyReturnsMean / dailyReturnsStD
    sharpeRatioAnnualized = k * sharpeRatio
    # print(sharpeRatioAnnualized)


if __name__ == "__main__":
    startvalue = 100000
    allocs = [0.3, 0.3, 0.3, 0.1]

    endDate = "2016-01-01"
    startDate = "2015-01-01"
    ##startdate end date in context of portfolio
    df1 = readData("data/TC1-RELIANCE.csv", startDate, endDate)
    df2 = readData("data/TC1-HDFCBANK.csv", startDate, endDate)
    df3 = readData("data/TC1-SUNPHARMA.csv", startDate, endDate)
    df4 = readData("data/TC1-ONGC.csv", startDate, endDate)
    # df1 = readData("data/NSE-RELIANCE.csv")
    # df2 = readData("data/NSE-HDFCBANK.csv")
    # df3 = readData("data/NSE-SUNPHARMA.csv")
    # df4 = readData("data/NSE-ONGC.csv")


    frames = [df1, df2, df3, df4]

    allStocks = pandas.concat(frames, keys=['REL', 'HDFC', 'SUN', 'ONGC'], axis=1)
    print(allStocks)
    plotter.plot(allStocks)
    # plotter.show()

    optimizePortfolio(allocs, startvalue, allStocks)