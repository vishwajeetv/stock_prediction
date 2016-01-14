import re
import csv

from itertools import count
import requests

HEADERS = {'user-agent': 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'}
URL = "http://www.example.com/img%03d.png"

# with a session, we get keep alive
session = requests.session()

with open('data/NSE_datasets-codes-cleaned.csv', 'r') as inp, open('data/all-stocks.csv', 'wb') as out:
    # writer = csv.writer(out)
    for row in csv.reader(inp):
            full_url = "https://www.quandl.com/api/v3/datasets/"+row[0]+".csv?auth_token=qPvscoNKm8eS7cjipW7k&collapse=quarterly"
            ignored, filename = URL.rsplit('/', 1)

            response = session.get(full_url, headers=HEADERS)
            if not response.ok:
                break
            out.write(response.content)


