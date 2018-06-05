import requests
import csv

def read_file():
    with open("4pp.csv") as f:
        reader = csv.reader(f)
        next(reader) # skip header
        data = [r for r in reader]
    return(data)

def get_LL(postalCode,data) :
    for i in data:
        if i[1] == postalCode:
            return [i[-2],i[-3]]

LL = get_LL("2152",read_file())
secret_key = ""

longitude = LL[0]
latitude = LL[1]

URL = "https://api.darksky.net/forecast/" + secret_key + "/" + str(longitude) + "," + str(latitude)
r = requests.get(URL)
print(r.content)
