import planetary_computer
import pystac_client
import rasterio
import tifffile
import urllib.request
import io

catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
search = catalog.search(
    collections=["sentinel-1-grd"],
    bbox=[10.95, 49.38, 11.20, 49.52],
    datetime="2023-09-01/2023-10-31",
)
items = list(search.items())
if not items:
    print("No items found")
    exit(1)

for item in items:
    item_signed = planetary_computer.sign(item)
    vv_url = item_signed.assets["vv"].href
    
    req = urllib.request.Request(vv_url, headers={"Range": "bytes=0-500000"})
    with urllib.request.urlopen(req) as response:
        header_data = response.read()

    with tifffile.TiffFile(io.BytesIO(header_data)) as tif:
        compressions = []
        for page in tif.pages:
            for tag in page.tags.values():
                print(f"Tag {tag.code} ({tag.name}): {tag.value}")
            break
    break
