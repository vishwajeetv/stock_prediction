import re
import csv

with open('data/NSE-datasets-codes.csv', 'r') as inp, open('data/NSE_datasets-codes-cleaned.csv', 'w') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if "NSE/CNX" not in row[0] and "NSE/NIFTY" not in row[0] and "NSE/SPCNX" not in row[0]:
            writer.writerow(row)

with open('data/all-stocks.csv', 'r',-1,"utf-8") as inp, open('data/all-stocks-cleaned.csv', 'w') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if "Date" not in row[0]:
            writer.writerow(row)