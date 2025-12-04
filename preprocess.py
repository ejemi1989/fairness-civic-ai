import pandas as pd

# Load GDELT
gdelt_df = pd.read_csv("data/GDELT_sample.csv", sep="\t", encoding='utf-8', usecols=["SQLDATE","EventRootCode","Actor1Name","Actor2Name","ActionGeo_FullName"])
gdelt_df.rename(columns={"ActionGeo_FullName":"region","SQLDATE":"date","EventRootCode":"label","Actor1Name":"text"}, inplace=True)

# Load NYC Open Data
nyc_df = pd.read_csv("data/NYC_311_Service_Requests.csv", usecols=["Complaint Type","Descriptor","Borough"])
nyc_df.rename(columns={"Complaint Type":"text","Borough":"region","Descriptor":"label"}, inplace=True)

# Load Chicago Data
chi_df = pd.read_csv("data/Chicago_Crime.csv", usecols=["Primary Type","Description","Community Area"])
chi_df.rename(columns={"Description":"text","Community Area":"region","Primary Type":"label"}, inplace=True)

# Combine datasets
df = pd.concat([gdelt_df, nyc_df, chi_df], ignore_index=True)

# Basic preprocessing
df['text'] = df['text'].astype(str).str.lower().str.replace(r'\W+', ' ', regex=True)
df['label'] = df['label'].astype(str)

# Save preprocessed data
df.to_csv("data/combined_civic.csv", index=False)
print("Preprocessing complete, combined dataset saved.")
