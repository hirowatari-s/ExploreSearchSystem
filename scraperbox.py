import urllib.parse
import urllib.request
import ssl
import json
import os
import pandas as pd


API_TOKEN = os.environ.get("SCRAPER_BOX_TOKEN")


ssl._create_default_https_context = ssl._create_unverified_context


def fetch_gsearch_result(search_str):
    # Urlencode the query string
    q = urllib.parse.quote_plus(search_str)

    # Create the query URL.
    query = "https://api.scraperbox.com/google"
    query += "?token=%s" % API_TOKEN
    query += "&q=%s" % q
    query += "&proxy_location=jp"
    query += "&results=100"
    print("query:", query)

    # Call the API.
    request = urllib.request.Request(query)
    print("Request start")
    raw_response = urllib.request.urlopen(request).read()
    print("Request end")
    raw_json = raw_response.decode("utf-8")
    response = json.loads(raw_json)

    # Print the first result title
    print(response)
    print(response["organic_results"][0]["title"])


    results = response["organic_results"]

    site_names = [res["title"] for res in results]
    num_items = len(site_names)
    keywords = [search_str] * num_items
    urls = [res["link"] for res in results]
    snippets = [res["snippet"] for res in results]
    rankings = list(range(1, num_items+1))

    df = pd.DataFrame(data=dict(
        keyword=keywords,
        site_name=site_names,
        URL=urls,
        snippet=snippets,
        ranking=rankings,
    ))
    return df


if __name__ == '__main__':
    search_str = input()
    df = fetch_gsearch_result(search_str)
    df.to_csv(search_str + ".csv")

