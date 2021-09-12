import urllib.request
from urllib.parse import urlparse

url = 'https://google.com/search?q=ファッション&filter=0&num=100'

# Perform the request
p = urlparse(url)
query = urllib.parse.quote_plus(p.query, safe='=&')
url = '{}://{}{}{}{}{}{}{}{}'.format(
    p.scheme, p.netloc, p.path,
    ';' if p.params else '', p.params,
    '?' if p.query else '', query,
    '#' if p.fragment else '', p.fragment)
request = urllib.request.Request(url)

# Set a normal User Agent header, otherwise Google will block the request.
request.add_header('User-Agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36')
raw_response = urllib.request.urlopen(request).read()

# Read the repsonse as a utf-8 string
html = raw_response.decode("utf-8")

from bs4 import BeautifulSoup

# The code to get the html contents here.

soup = BeautifulSoup(html, 'html.parser')

# Find all the search result divs
divs = soup.select("#search div.g")

for div in divs:
    # Search for a h3 tag
    results = div.select("h3")
    if (len(results) >= 1):
        # Print the title
        print(results[0].get_text())

    # Search for a span tag
    results = div.select("span")
    if (len(results) >= 1):
        # Print the snnipet
        print("\t",  results[-1].get_text())

    print()


# Print number of search result
print("count:", len(divs))
