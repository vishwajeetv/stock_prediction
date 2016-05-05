import Quandl
import numpy as np
import array
import matplotlib.pyplot as plotter
import pandas
import csv
import math
import scipy.optimize as optimizer

def readData(file):
    prices = pandas.read_csv(file, parse_dates=True, index_col='Date', usecols=['Date', 'Close']).fillna(method='ffill')
    return prices

def normalize(df):
    return df/df1['Close'][0]

df1 = readData("data/NSE-RELIANCE.csv")
df2 = readData("data/NSE-HDFCBank.csv")
df3 = readData("data/NSE-SUNPHARMA.csv")
df4 = readData("data/NSE-ONGC.csv")

frames = [df1, df2, df3, df4]

allStocks = pandas.concat(frames, keys=['REL', 'HDFC', 'SUN','ONGC'], axis=1)
allStocks = allStocks.ix['2015-01-01':'2016-01-01']

plotter.plot(allStocks)
# plotter.show()

def optimizeCumulativeReturns(allocs, allStocksNormalized):
    allocs = allStocksNormalized * allocs
    postvalues = allocs * 100000
    portfolioValues = postvalues.sum(axis=1)
    cumulativeReturns = (portfolioValues[-1] / portfolioValues[0]) - 1
    return cumulativeReturns*-1

def optimizeSharpe(allocs, allStocksNormalized):
    allocs = allStocksNormalized * allocs
    postvalues = allocs * 100000
    portfolioValues = postvalues.sum(axis=1)

    dailyReturns = (portfolioValues / portfolioValues.shift(1)) - 1
    dailyReturns = dailyReturns[1:]

    dailyReturnsStD = dailyReturns.std()
    dailyReturnsMean = dailyReturns.mean()
    k = math.sqrt(252)
    sharpeRatio = dailyReturnsMean / dailyReturnsStD
    sharpeRatioAnnualized = k * sharpeRatio
    return sharpeRatioAnnualized*-1

def optimizePortfolio(allocs, startvalue):
    allStocksNormalized = normalize(allStocks)
    # plotter.plot(allStocksNormalized)
    # plotter.show()


    allocsGuess = [0.3,0.3,0.3,0.1]

    cons = ({'type': 'eq', 'fun' : lambda inputs: 1 - (np.sum(np.absolute(inputs)))})
    bnds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]

    #optimize for cumulative returns
    optimizedOutput = optimizer.minimize(optimizeCumulativeReturns, allocsGuess, args=(allStocksNormalized,) ,
                                       method='SLSQP',  bounds=bnds, constraints=cons,
                                        options={'disp': True})

    #optimize for sharpe ratio
    # optimizedOutput = optimizer.minimize(optimizeSharpe, allocsGuess, args=(allStocksNormalized,) ,
    #                                        method='SLSQP',  bounds=bnds, constraints=cons,
    #                                         options={'disp': True})

    # print("X = {}, Y = {}".format(optimizedOutput.x,optimizedOutput.fun))
    print("REL :"+'{:f}'.format(optimizedOutput.x[0]))
    print("HDFC :"+'{:f}'.format(optimizedOutput.x[1]))
    print("SUN :"+'{:f}'.format(optimizedOutput.x[2]))
    print("ONGC :"+'{:f}'.format(optimizedOutput.x[3]))

    optimizedAllocations = optimizedOutput.x
    allocationsApplied = allStocksNormalized * optimizedAllocations
    postvalues = allocationsApplied * startvalue
    portfolioValues = postvalues.sum(axis=1)

    print(portfolioValues.tail())
    cumulativeReturns = (portfolioValues[-1] / portfolioValues[0]) - 1
    print(cumulativeReturns)
    plotter.plot(portfolioValues)
    plotter.show()
    dailyReturns = (portfolioValues / portfolioValues.shift(1)) - 1
    dailyReturns = dailyReturns[1:]

    dailyReturnsStD = dailyReturns.std()
    dailyReturnsMean = dailyReturns.mean()
    k = math.sqrt(252)
    sharpeRatio = dailyReturnsMean / dailyReturnsStD
    sharpeRatioAnnualized = k * sharpeRatio
    # print(sharpeRatioAnnualized)

startvalue = 1000000
allocs = [0.3,0.3,0.3,0.1]

optimizePortfolio(allocs, startvalue)

