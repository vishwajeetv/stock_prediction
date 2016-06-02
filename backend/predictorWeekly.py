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
import sklearn.ensemble as ensemble

def readData():
    file = "data/TC1-ONGC.csv"
    # reader = pandas.read_csv("data/TATAMOTORS/NSE-TATAMOTORS.csv",index_col='Date',parse_dates = True )
    df = pandas.read_csv(file, parse_dates=True, index_col='Date', usecols=['Date', 'Close Price']).fillna(
        method='ffill')
    return df

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
    simpleMovingAverage = simpleMovingAverage.dropna()
    return simpleMovingAverage

def mergeAll(df):
    changeInPrice = calculateChangeInPrice(df)
    bollingerBandsValue = calculateBollingerBandsValue(df)
    simpleMovingAverage = calculateSimpleMovingAverage(df)
    dfMerged = df.merge(changeInPrice, how='inner', left_index=True, right_index=True, suffixes=('', ' PtChange'))
    dfMerged = dfMerged.merge(bollingerBandsValue, how='inner', left_index=True, right_index=True, suffixes=('', ' BBVal'))
    dfMerged = dfMerged.merge(simpleMovingAverage, how='inner', left_index=True, right_index=True, suffixes=('', ' SMA'))
    return dfMerged

def getLearnableData(df):
    bbVal = df[['Close Price BBVal']]
    bbVal = np.reshape(bbVal, (bbVal.size, 1))

    sma = df[['Close Price SMA']]
    sma = np.reshape(sma, (sma.size, 1))

    # dataX = bbVal
    dataX = bbVal.merge(sma, how='inner', left_index=True, right_index=True, suffixes=('', ''))

    ptChange = df[['Close Price PtChange']]
    ptChange = np.reshape(ptChange, (ptChange.size, 1))

    dataY = ptChange

    return dataX, dataY

def sample(df):
    dfTrain = df.ix["2015-05-01":"2012-05-01"]
    dfTrainMerged = mergeAll(dfTrain)
    dfTest = df.ix["2016-05-01":"2015-05-01"]
    dfTestMerged = mergeAll(dfTest)
    dataTrainX, dataTrainY = getLearnableData(dfTrainMerged)
    dataTestX, dataTestY = getLearnableData(dfTestMerged)
    return dataTrainX, dataTrainY, dataTestX, dataTestY

def adbPredictor(df):
    dataTrainX, dataTrainY, dataTestX, dataTestY = sample(df)

    # clf = linear_model.SGDRegressor()
    clf = ensemble.AdaBoostRegressor()
    clf.fit(dataTrainX, dataTrainY)

    predicted = clf.predict(dataTestX)

    fig, ax = plotter.subplots()

    ax.set_ylabel('Predicted KNN Weekly')
    ax.scatter(dataTestY, predicted)
    ax.set_xlabel('Measured')
    predicted = np.reshape(predicted, (predicted.size, 1))
    corrCoeff = pearsonr(dataTestY,predicted)
    print(corrCoeff[0])
    plotter.show()
    return predicted

def knnPredictor(df):

    dataTrainX, dataTrainY, dataTestX, dataTestY = sample(df)
    corelationCoefficiantDictionary = {}
    corelationCoefficiantArray = []

    for k in range(1, 200, 1):
        knnModel = KNeighborsRegressor(n_neighbors=k)

        knnModel.fit(dataTrainX, dataTrainY)

        knnpredicted = knnModel.predict(dataTestX)
        corelationCoefficient = pearsonr(dataTestY, knnpredicted)
        corelationCoefficiantDictionary[k] = corelationCoefficient[0]
        corelationCoefficiantArray.append(corelationCoefficient[0])

    # plotter.plot(corelationCoefficiantArray)
    bestK = max(corelationCoefficiantDictionary, key=corelationCoefficiantDictionary.get)

    knnModelBest = KNeighborsRegressor(n_neighbors=bestK)
    knnModelBest.fit(dataTrainX, dataTrainY)
    print("K = ")
    print(bestK)
    print("Corelation Coeff:")
    print(corelationCoefficiantDictionary[bestK])

    knnpredictedBest = knnModelBest.predict(dataTestX)

    fig, ax = plotter.subplots()
    corelationCoefficient = pearsonr(dataTestY, knnpredictedBest)
    print(corelationCoefficient[0])
    ax.set_ylabel('Predicted KNN Weekly')
    ax.scatter(dataTestY, knnpredictedBest)
    ax.set_xlabel('Measured')
    plotter.show()




def randomForestPredictor(df):

    # bbValTest, bbValTrain, ptChangeTest, ptChangeTrain = sample(df)
    dataTrainX, dataTrainY, dataTestX, dataTestY = sample(df)
    corelationCoefficiantDictionary = {}
    corelationCoefficiantArray = []

    for k in range(1, 100, 1):
        rfsModel = RandomForestRegressor(n_estimators=k)
        rfsModel.fit(dataTrainX, dataTrainY)

        rfspredicted = rfsModel.predict(dataTestX)
        rfspredicted = np.reshape(rfspredicted, (rfspredicted.size, 1))
        corelationCoefficient = pearsonr(dataTestY, rfspredicted)
        corelationCoefficiantDictionary[k] = corelationCoefficient[0]
        corelationCoefficiantArray.append(corelationCoefficient[0])

    plotter.plot(corelationCoefficiantArray)
    # plotter.show()
    bestK = max(corelationCoefficiantDictionary, key=corelationCoefficiantDictionary.get)

    rfsModelBest = RandomForestRegressor(n_estimators=bestK)

    rfsModelBest.fit(dataTrainX, dataTrainY)
    print("K = ")
    print(bestK)
    print("Correlation Coefficient =")
    print(corelationCoefficiantDictionary[bestK])
    rfsPredictedBest = rfsModelBest.predict(dataTestX)

    fig, ax = plotter.subplots()

    ax.set_ylabel('Predicted RandomForest Weekly')
    ax.scatter(dataTestY, rfsPredictedBest)
    ax.set_xlabel('Measured')
    plotter.show()



if __name__ == "__main__":

    df = readData()

    knnPredictor(df)
    # randomForestPredictor(df)
    # adbPredictor(df)


    # priceToPredict = 1185.90

    # bollingerBandsValue = calculateBollingerBandsValue(dfTest)
    # print(bollingerBandsValue)


