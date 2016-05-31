import Quandl
import numpy as np
import array
from sklearn import datasets,linear_model, svm, neural_network
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plotter
import pandas
import csv
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor

def readData():
    # stock = Quandl.get("NSE/INFY")
    # stock = np.array(stock)
    # currentRatio = Quandl.get("DEB/INFY_A_CRATIO")
    # reader = pandas.io.parsers.read_csv("data/all-stocks-cleaned.csv")
    file = "data/TC1-HDFCBANK.csv"
    # reader = pandas.read_csv("data/TATAMOTORS/NSE-TATAMOTORS.csv",index_col='Date',parse_dates = True )
    df = pandas.read_csv(file, parse_dates=True, index_col='Date',
                         usecols=['Date', 'Close Price'])
    df = df.fillna(method='ffill')
    # df = df.dropna()
    dfAll = pandas.read_csv(file)
    dfAll = dfAll.fillna(method='ffill')
    stock = np.array(dfAll)
    # plotter.xlabel('Date')
    # plotter.ylabel('Close')
    return df, stock


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


def sample(stock):
    openingPrice = stock[:, 1]
    closingPrice = stock[:, 5]
    openingPriceTrain, openingPriceTest, closingPriceTrain, closingPriceTest = \
        train_test_split(openingPrice, closingPrice, test_size=0.25, random_state=42)
    openingPriceTrain = np.reshape(openingPriceTrain, (openingPriceTrain.size, 1))
    closingPriceTrain = np.reshape(closingPriceTrain, (closingPriceTrain.size, 1))
    openingPriceTest = np.reshape(openingPriceTest, (openingPriceTest.size, 1))
    closingPriceTest = np.reshape(closingPriceTest, (closingPriceTest.size, 1))

    sampledData = {"openingPriceTrain":openingPriceTrain, "closingPriceTrain":closingPriceTrain,
                   "openingPriceTest":openingPriceTest, "closingPriceTest":closingPriceTest}
    return sampledData

def predictRandomForestReg(data, priceToPredict):
    openingPriceTrain, openingPriceTest, closingPriceTrain, closingPriceTest = \
        data["openingPriceTrain"], data["openingPriceTest"], data["closingPriceTrain"], data["closingPriceTest"]
    k = 10
    clf = RandomForestRegressor(n_estimators=k)
    clf = clf.fit(openingPriceTrain, closingPriceTrain)
    openingPriceToPredict = np.array([priceToPredict])
    print(clf.predict(openingPriceToPredict))

def predictKnn(data, priceToPredict):
    corelationCoefficiantDictionary = {}
    corelationCoefficiantArray = []
    openingPriceTrain, openingPriceTest, closingPriceTrain, closingPriceTest = \
        data["openingPriceTrain"], data["openingPriceTest"], data["closingPriceTrain"], data["closingPriceTest"]

    for k in range( 1 , 100 , 1):
        neigh = KNeighborsRegressor(n_neighbors=k)
        #n = 7 best fits
        neigh.fit(openingPriceTrain, closingPriceTrain)

        closingPriceTestArray = np.reshape(closingPriceTest,-1)
        knnpr = neigh.predict(openingPriceTest)
        predictedArray = np.reshape(knnpr,-1)

        corelationCoefficient = pearsonr(closingPriceTestArray,predictedArray)
        corelationCoefficiantDictionary[k] = corelationCoefficient[0]
        corelationCoefficiantArray.append(corelationCoefficient[0])
    plotter.plot(corelationCoefficiantArray)
    # plotter.show()

    bestK = max(corelationCoefficiantDictionary, key=corelationCoefficiantDictionary.get)
    neighBest = KNeighborsRegressor(n_neighbors=bestK)
    neighBest.fit(openingPriceTrain, closingPriceTrain)
    openingPriceToPredict = np.array([priceToPredict])
    print("K = ")
    print(bestK)
    print(neighBest.predict(openingPriceToPredict))


def predict(data, priceToPredict):

    openingPriceTrain, openingPriceTest, closingPriceTrain, closingPriceTest = \
        data["openingPriceTrain"], data["openingPriceTest"], data["closingPriceTrain"], data["closingPriceTest"]
    clf = svm.LinearSVR()
    clf.fit(openingPriceTrain, closingPriceTrain)
    predicted2 = clf.predict(openingPriceTest)
    score = clf.fit(openingPriceTrain, closingPriceTrain).score(openingPriceTest, closingPriceTest)
    # print(score)

    fig, ax = plotter.subplots()
    ax.scatter(openingPriceTrain, closingPriceTrain)
    ax.set_ylabel('Predicted SVM')
    ax.scatter(closingPriceTest, clf.predict(openingPriceTest))
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    # plotter.show()

    closingPriceTestArray = np.reshape(closingPriceTest,-1)
    clfpr = clf.predict(openingPriceTest)
    predictedArray = np.reshape(clfpr,-1)
    print(pearsonr(closingPriceTestArray,predictedArray))

    openingPriceToPredict = np.array([priceToPredict])
    print(clf.predict(openingPriceToPredict))
    return clf.predict(np.array([openingPriceToPredict]))

def calculateDailyReturns(df):
    dailyReturns = (df/df.shift(1)) - 1
    # dailyReturns.ix[0,:] = 0
    # cumulativeReturns =  (df/df['Close'][0]) - 1
    # dailyReturns = df['Close'].pct_change(1)
    plotter.plot(dailyReturns)
    # dailyReturns.hist(bins=100)
    plotter.show()


if __name__ == "__main__":

    df, stock = readData()

    # plot(df)
    df = df.ix[500:1000,:]
    # calculateDailyReturns(df)
    sampledData = sample(stock)
    priceToPredict = 1185.90
    print("SVM Prediction:")
    predict(sampledData, priceToPredict)
    print("KNN Prediction:")
    predictKnn(sampledData, priceToPredict)
    print("Random Forest Regressor Prediction:")
    predictRandomForestReg(sampledData, priceToPredict)

