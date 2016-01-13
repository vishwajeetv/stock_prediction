import re
import csv
with open('data/NSE-datasets-codes.csv', 'r') as inp, open('data/NSE_datasets-codes-cleaned.csv', 'w') as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if ~bool(re.search('CNX', row[1])):
            writer.writerow(row)