import Quandl
import numpy as np
import array
from sklearn import datasets,linear_model, svm, neural_network
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plotter
import pandas
import csv


def readData():
    # stock = Quandl.get("NSE/INFY")
    # stock = np.array(stock)
    # currentRatio = Quandl.get("DEB/INFY_A_CRATIO")
    # reader = pandas.io.parsers.read_csv("data/all-stocks-cleaned.csv")
    # reader = pandas.read_csv("data/TATAMOTORS/NSE-TATAMOTORS.csv",index_col='Date',parse_dates = True )
    df = pandas.read_csv("data/TATAMOTORS/NSE-TATAMOTORS.csv", parse_dates=True, index_col='Date',
                         usecols=['Date', 'Close'])
    df = df.fillna(method='ffill')
    # df = df.dropna()
    dfAll = pandas.read_csv("data/TATAMOTORS/NSE-TATAMOTORS.csv", parse_dates=True, index_col='Date')
    dfAll = dfAll.fillna(method='ffill')
    stock = np.array(dfAll)
    # plotter.xlabel('Date')
    # plotter.ylabel('Close')
    return df, stock


def plot(df):
    rollingMean = pandas.rolling_mean(df['Close'], window=100)
    rollingStdv = pandas.rolling_std(df['Close'], window=100)
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


def predict(data):

    openingPriceTrain, openingPriceTest, closingPriceTrain, closingPriceTest = \
        data["openingPriceTrain"], data["openingPriceTest"], data["closingPriceTrain"], data["closingPriceTest"]
    clf = svm.LinearSVR()
    clf.fit(openingPriceTrain, closingPriceTrain)
    predicted2 = clf.predict(openingPriceTest)
    score = clf.fit(openingPriceTrain, closingPriceTrain).score(openingPriceTest, closingPriceTest)
    print(score)
    fig, ax = plotter.subplots()
    ax.scatter(openingPriceTrain, closingPriceTrain)
    ax.scatter(closingPriceTest, clf.predict(openingPriceTest))
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plotter.show()
    openingPriceToPredict = np.array([335.50])
    print(clf.predict(openingPriceToPredict))
    return clf.predict(np.array([openingPriceToPredict]))


if __name__ == "__main__":

    df, stock = readData()

    # plot(df)

    sampledData = sample(stock)

    predict(sampledData)
