

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
headers = {"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36",
           }

url = r'https://steve-p.org/cards/RWSa.html'
url_half = url.rsplit("/",1)[0]

site = requests.get(url, headers = headers)
soup = BeautifulSoup(site.text, 'html.parser')
image_url_list = soup.find_all("img", {"class":"partofdeck card"})

for it in tqdm(image_url_list):
    src = it["src"]
    image_url = url_half + "/" + src

    save_path = f'./TarotImg/{src.replace("small/sm_","")}'
    file = requests.get(image_url)
    with open(save_path, "wb") as f:
        f.write(file.content)