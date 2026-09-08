import ee
import json

def initialize_earth(credentials_file: Path):
    with open(credentials_file, "r") as fil:
        creds = json.load(fil)
    service_account = creds["client_email"]
    credentials = ee.ServiceAccountCredentials(service_account, str(credentials_file))
    ee.Initialize(credentials, project=creds["project_id"])
    
