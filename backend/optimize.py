import Quandl
import numpy as np
import array
import matplotlib.pyplot as plotter
import pandas
import csv

def readData(file):
    return pandas.read_csv(file, parse_dates=True, index_col='Date', usecols=['Date', 'Close']).fillna(method='ffill')

def normalize(df):
    return df/df1['Close'][0]

df1 = readData("data/NSE-RELIANCE.csv")
df2 = readData("data/NSE-HDFCBank.csv")
df3 = readData("data/NSE-SUNPHARMA.csv")
df4 = readData("data/NSE-ONGC.csv")


# plotter.plot(df1)
# plotter.plot(df2)
# plotter.plot(df3)
# plotter.plot(df4)
# plotter.show()

# print(df1['Close'][0])
start = 1000000
allocs = [0.2,0.2,0.2,0.2]

df1n = normalize(df1)
df2n = normalize(df2)
df3n = normalize(df3)
df4n = normalize(df4)

plotter.plot(df1n)
plotter.plot(df2n)
plotter.plot(df3n)
plotter.plot(df4n)
plotter.show()