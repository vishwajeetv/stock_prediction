import re
import csv
import math
import numpy
from numbers import Number
# with open('data/NSE-datasets-codes.csv', 'r') as inp, open('data/NSE_datasets-codes-cleaned.csv', 'w') as out:
#     writer = csv.writer(out)
#     for row in csv.reader(inp):
#         if "NSE/CNX" not in row[0] and "NSE/NIFTY" not in row[0] and "NSE/SPCNX" not in row[0]:
#             writer.writerow(row)

with open('data/all-stocks.csv', 'r',1,"utf-8") as inp, open('data/all-stocks-cleaned.csv', 'w') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if "Date" not in row[0]:
            if str(row[1]) != 'nan':
                 if len(row) > 5:
                     if str(row[5]) != 'nan':
                            writer.writerow(row)


