import Quandl
import numpy as np
import array
from sklearn import datasets,linear_model, svm, cross_validation, neural_network
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plotter
import pandas
import csv

# stock = Quandl.get("NSE/INFY")
# stock = np.array(stock)
# currentRatio = Quandl.get("DEB/INFY_A_CRATIO")

reader = pandas.io.parsers.read_csv("data/all-stocks-cleaned.csv")
reader = reader.fillna(method='ffill')
stock = np.array(reader)

openingPrice = stock[:, 1]
closingPrice = stock[:, 5]

print(np.max(closingPrice))

openingPriceTrain, openingPriceTest, closingPriceTrain, closingPriceTest = \
    train_test_split(openingPrice, closingPrice, test_size=0.25, random_state=42)


openingPriceTrain = np.reshape(openingPriceTrain,(openingPriceTrain.size,1))



closingPriceTrain = np.reshape(closingPriceTrain,(closingPriceTrain.size,1))


openingPriceTest = np.reshape(openingPriceTest,(openingPriceTest.size,1))
closingPriceTest = np.reshape(closingPriceTest,(closingPriceTest.size,1))


# regression = linear_model.LinearRegression()
#
# regression.fit(openingPriceTrain, closingPriceTrain)
#
#
# predicted = regression.predict(openingPriceTest)
#
# plotter.scatter(openingPriceTest, closingPriceTest,  color='black')
# plotter.plot(closingPriceTest, regression.predict(openingPriceTest))
# plotter.show()


clf = svm.LinearSVR()
clf.fit(openingPriceTrain, closingPriceTrain)
predicted2 = clf.predict(openingPriceTest)
score = clf.fit(openingPriceTrain, closingPriceTrain).score(openingPriceTest,closingPriceTest)
print(score)
print(cross_validation.cross_val_score(clf,openingPriceTest,closingPriceTest))
# plotter.scatter(openingPriceTest, closingPriceTest,  color='black')
plotter.plot(closingPriceTest, clf.predict(openingPriceTest))
plotter.show()


# mlpr = neural_network.MLPRegressor(10)
# mlpr.fit(openingPriceTrain, closingPriceTrain)
# predicted2 = mlpr.predict(openingPriceTest)
# # score = mlpr.fit(openingPriceTrain, closingPriceTrain).score(openingPriceTest,closingPriceTest)
# # print(score)
# # print(cross_validation.cross_val_score(clf,openingPriceTest,closingPriceTest))
# plotter.scatter(openingPriceTest, closingPriceTest,  color='black')
# plotter.plot(closingPriceTest, mlpr.predict(openingPriceTest))
# plotter.show()

