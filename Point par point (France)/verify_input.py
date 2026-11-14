from pathlib import Path
from urllib.request import urlopen
from html.parser import HTMLParser
import hashlib

ROOT = Path(__file__).resolve().parent
class Links(HTMLParser):
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if(tag == "a" and "href" in attrs): print(attrs["href"], flush=True)

page = urlopen("https://drive.google.com/embeddedfolderview?id=1WCd-8ZlYk-SCskQAqr4UT6I27I8KoNkk").read().decode()
Links().feed(page)
data = urlopen("https://drive.google.com/uc?export=download&id=1IrbgRIZhyvdVR_RsOzy6T71hKZitaXd9").read()
print("Remote bytes:", len(data), "SHA256:", hashlib.sha256(data).hexdigest())
local = (ROOT / "paris_listings.csv").read_bytes()
print("Local bytes:", len(local), "SHA256:", hashlib.sha256(local).hexdigest())
if(data != local): (ROOT / "source_paris_listings.csv").write_bytes(data)
