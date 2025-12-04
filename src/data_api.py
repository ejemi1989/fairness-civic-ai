import requests
import pandas as pd

# Chicago 311
CHICAGO_URL = "https://data.cityofchicago.org/api/v3/views/ijzp-q8t2/query.json"
def fetch_chicago(limit=1000):
    r = requests.get(CHICAGO_URL, params={"$limit": limit})
    data = pd.json_normalize(r.json()['data'])
    data.columns = [str(c) for c in data.columns]
    return data[['unique_key', 'service_request_type', 'status', 'community_area']]

# NYC 311
NYC_URL = "https://data.cityofnewyork.us/api/v3/views/erm2-nwe9/query.json"
def fetch_nyc(limit=1000):
    r = requests.get(NYC_URL, params={"$limit": limit})
    data = pd.json_normalize(r.json()['data'])
    return data[[0,1,2,5]]  # adjust columns as needed

if __name__ == "__main__":
    chicago = fetch_chicago(500)
    nyc = fetch_nyc(500)
    civic_df = pd.concat([chicago, nyc], ignore_index=True)
    civic_df.to_csv("data/api_civic.csv", index=False)
