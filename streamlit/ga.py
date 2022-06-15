from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.discovery import build
import pandas as pd
import httplib2
import os

# Get GA view-id for development or production environments
HOME = os.environ.get('HOME')
VIEW_ID = "269378613" if HOME == "/home/debian" else "269246028"

# Create service credentials
# Rename your JSON key to client_secrets.json and save it to your working folder
credentials = ServiceAccountCredentials.from_json_keyfile_name(
    'client_secrets.json', ['https://www.googleapis.com/auth/analytics.readonly'])

# Create a service object
http = credentials.authorize(httplib2.Http())
service = build('analytics', 'v4', http=http, discoveryServiceUrl=(
    'https://analyticsreporting.googleapis.com/$discovery/rest'))
response = service.reports().batchGet(
    body={
        "reportRequests": [
            {
                "viewId": VIEW_ID,
                "dateRanges": [{"startDate": "30daysAgo", "endDate": "today"},         {
                    "startDate": "365daysAgo",
                    "endDate": "today"
                }],
                # "metrics": [{"expression": "ga:users"}],
                # "dimensions": [{"name": "ga:pagePath"}],
                # "pageSize": 100
            }]
    }
).execute()

# create two empty lists that will hold our dimentions and sessions data
period = ['Last 7 days', 'Last 30 days']
users = []


# # Extract Data
for report in response.get('reports', []):
    columnHeader = report.get('columnHeader', {})
    dimensionHeaders = columnHeader.get('dimensions', [])
    metricHeaders = columnHeader.get(
        'metricHeader', {}).get('metricHeaderEntries', [])
    rows = report.get('data', {}).get('rows', [])

    for row in rows:

        dateRangeValues = row.get('metrics', [])

        for i, values in enumerate(dateRangeValues):
            for metricHeader, value in zip(metricHeaders, values.get('values')):
                users.append(int(value))


def get_ga_values():

    df = pd.DataFrame()
    df["Users"] = users
    df["Visitors"] = period
    df = df[["Visitors", "Users"]]

    # Display DataFrame without index
    blankIndex = [''] * len(df)
    df.index = blankIndex
    return df
