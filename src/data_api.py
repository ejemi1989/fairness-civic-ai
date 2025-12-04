import requests
import pandas as pd

# Chicago 311 API
CHICAGO_URL = "https://data.cityofchicago.org/api/v3/views/ijzp-q8t2/query.json"
def fetch_chicago(limit=500):
    r = requests.get(CHICAGO_URL, params={"$limit": limit})
    data = pd.json_normalize(r.json()['data'])
    return data[[0, 8, 9]]  # example columns: unique key, service type, community area

# NYC 311 API
NYC_URL = "https://data.cityofnewyork.us/api/v3/views/erm2-nwe9/query.json"
def fetch_nyc(limit=500):
    r = requests.get(NYC_URL, params={"$limit": limit})
    data = pd.json_normalize(r.json()['data'])
    return data[[0,1,5]]  # adjust column indices

if __name__ == "__main__":
    chicago = fetch_chicago()
    nyc = fetch_nyc()

    civic_data = pd.concat([chicago, nyc], ignore_index=True, sort=False)
    print(civic_data.head())
