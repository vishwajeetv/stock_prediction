import Quandl
import numpy as np
import array
from sklearn import datasets,linear_model, svm, neural_network
import matplotlib.pyplot as plotter
import pandas
import csv
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
import datetime as dt

def readData():
    file = "data/TC1-RELIANCE.csv"
    # reader = pandas.read_csv("data/TATAMOTORS/NSE-TATAMOTORS.csv",index_col='Date',parse_dates = True )
    df = pandas.read_csv(file, parse_dates=True, index_col='Date', usecols=['Date', 'Close Price']).fillna(
        method='ffill')
    return df


def plot(df):
    rollingMean = pandas.rolling_mean(df['Close Price'], window=100)
    rollingStdv = pandas.rolling_std(df['Close Price'], window=100)
    plotter.plot(rollingMean)
    # plotting bollinger bands
    plotter.plot(rollingMean + rollingStdv * 2)
    plotter.plot(rollingMean - rollingStdv * 2)
    # df1 = df[['Date','Close']]
    plotter.plot(df)
    plotter.show()

def calculateDailyReturns(df):
    dailyReturns = (df/df.shift(1)) - 1
    # dailyReturns.ix[0,:] = 0
    # cumulativeReturns =  (df/df['Close'][0]) - 1
    # dailyReturns = df['Close'].pct_change(1)
    plotter.plot(dailyReturns)
    # dailyReturns.hist(bins=100)
    plotter.show()

def normalize(df):
    normalizedValues = (df - df.mean())/df.std()
    return normalizedValues

def calculateChangeInPrice(df):
    changeInPrice = df.pct_change(5)
    changeInPrice = changeInPrice.dropna()
    return changeInPrice

def calculateBollingerBandsValue(df):
    simpleMovingAverage = pandas.rolling_mean(df,window=5)
    stdDeviation = pandas.rolling_std(df,window=5)
    bollingerBandsValue = (df - simpleMovingAverage)/(2*stdDeviation)
    bollingerBandsValue = bollingerBandsValue.dropna()
    return bollingerBandsValue

def calculateSimpleMovingAverage(df):
    simpleMovingAverage = pandas.rolling_mean(df,window=5)
    simpleMovingAverage = normalize(simpleMovingAverage)
    simpleMovingAverage = calculateSimpleMovingAverage.dropna()
    return simpleMovingAverage

def mergeAll(df):
    changeInPrice = calculateChangeInPrice(df)
    bollingerBandsValue = calculateBollingerBandsValue(df)
    simpleMovingAverage = calculateSimpleMovingAverage(df)
    dfMerged = df.merge(changeInPrice, how='inner', left_index=True, right_index=True, suffixes=('', ' PtChange'))
    dfMerged = dfMerged.merge(bollingerBandsValue, how='inner', left_index=True, right_index=True, suffixes=('', ' BBVal'))
    dfMerged = dfMerged.merge(simpleMovingAverage, how='inner', left_index=True, right_index=True, suffixes=('', ' BBVal'))
    return dfMerged

def getLearnableData(df):
    bbVal = df[['Close Price BBVal']]
    bbVal = np.reshape(bbVal, (bbVal.size, 1))

    ptChange = df[['Close Price PtChange']]
    ptChange = np.reshape(ptChange, (ptChange.size, 1))

    return bbVal, ptChange

def randomForestPredictor(df):

    bbValTest, bbValTrain, ptChangeTest, ptChangeTrain = sample(df)

    corelationCoefficiantDictionary = {}
    corelationCoefficiantArray = []

    for k in range(1, 100, 1):
        rfsModel = RandomForestRegressor(n_estimators=k)
        rfsModel.fit(bbValTrain, ptChangeTrain)

        rfspredicted = rfsModel.predict(bbValTest)
        rfspredicted = np.reshape(rfspredicted, (rfspredicted.size, 1))
        corelationCoefficient = pearsonr(bbValTest, rfspredicted)
        corelationCoefficiantDictionary[k] = corelationCoefficient[0]
        corelationCoefficiantArray.append(corelationCoefficient[0])

    plotter.plot(corelationCoefficiantArray)
    # plotter.show()
    bestK = max(corelationCoefficiantDictionary, key=corelationCoefficiantDictionary.get)

    rfsModelBest = RandomForestRegressor(n_estimators=bestK)

    rfsModelBest.fit(bbValTrain, ptChangeTrain)
    print("K = ")
    print(bestK)
    print("Correlation Coefficient =")
    print(corelationCoefficiantDictionary[bestK])
    rfsPredictedBest = rfsModelBest.predict(bbValTest)

    fig, ax = plotter.subplots()

    ax.set_ylabel('Predicted RandomForest Weekly')
    ax.scatter(ptChangeTest, rfsPredictedBest)
    ax.set_xlabel('Measured')
    plotter.show()


def sample(df):
    dfTrain = df.ix["2015-05-01":"2010-05-01"]
    dfTrainMerged = mergeAll(dfTrain)
    dfTest = df.ix["2016-05-01":"2015-05-01"]
    dfTestMerged = mergeAll(dfTest)
    bbValTrain, ptChangeTrain = getLearnableData(dfTrainMerged)
    bbValTest, ptChangeTest = getLearnableData(dfTestMerged)
    return bbValTest, bbValTrain, ptChangeTest, ptChangeTrain


def knnPredictor(df):

    bbValTest, bbValTrain, ptChangeTest, ptChangeTrain = sample(df)

    corelationCoefficiantDictionary = {}
    corelationCoefficiantArray = []

    for k in range(1, 200, 1):
        knnModel = KNeighborsRegressor(n_neighbors=k)
        knnModel.fit(bbValTrain, ptChangeTrain)

        knnpredicted = knnModel.predict(bbValTest)
        corelationCoefficient = pearsonr(bbValTest, knnpredicted)
        corelationCoefficiantDictionary[k] = corelationCoefficient[0]
        corelationCoefficiantArray.append(corelationCoefficient[0])

    # plotter.plot(corelationCoefficiantArray)
    bestK = max(corelationCoefficiantDictionary, key=corelationCoefficiantDictionary.get)

    knnModelBest = KNeighborsRegressor(n_neighbors=bestK)

    knnModelBest.fit(bbValTrain, ptChangeTrain)
    print("K = ")
    print(bestK)
    print("Corelation Coeff:")
    print(corelationCoefficiantDictionary[bestK])
    knnpredictedBest = knnModelBest.predict(bbValTest)

    fig, ax = plotter.subplots()

    ax.set_ylabel('Predicted KNN Weekly')
    ax.scatter(ptChangeTest, knnpredictedBest)
    ax.set_xlabel('Measured')
    plotter.show()

if __name__ == "__main__":

    df = readData()

    knnPredictor(df)
    # randomForestPredictor(df)


    # priceToPredict = 1185.90

    # bollingerBandsValue = calculateBollingerBandsValue(dfTest)
    # print(bollingerBandsValue)


