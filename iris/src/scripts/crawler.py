import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from PIL import Image
import imagehash
from io import BytesIO
import time

def crawl_for_matching_images(target_image_path, start_url, max_pages=10, tolerance_threshold=5):
    target_hash = None
    try:
        target_hash = imagehash.phash(Image.open(target_image_path))
    except Exception as e:
        print(f"Erreur lecture image: {e}")
        return []

    visited = set()
    queue = [start_url]
    domain = urlparse(start_url).netloc.replace('www.', '')
    matches = []

    while queue and len(visited) < max_pages:
        url = queue.pop(0)
        
        if url in visited:
            continue
            
        visited.add(url)
        print(f"scan : {url}")

        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept-Language': 'fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7'
            }
            resp = requests.get(url, headers=headers, timeout=10)
            
            if resp.status_code != 200:
                print(f"bloquer!! ({resp.status_code})")
                continue

            soup = BeautifulSoup(resp.text, 'html.parser')

            for img in soup.find_all('img'):
                src = img.get('src')
                if not src or src.endswith('.svg') or src.startswith('data:'):
                    continue

                full_url = urljoin(url, src)

                try:
                    img_resp = requests.get(full_url, headers=headers, timeout=5)
                    img_hash = imagehash.phash(Image.open(BytesIO(img_resp.content)))
                    
                    if (target_hash - img_hash) <= tolerance_threshold:
                        print(f"-> Trouvé: {full_url}")
                        matches.append(full_url)
                except:
                    pass

            for a in soup.find_all('a', href=True):
                next_url = urljoin(url, a['href'])
                if domain in urlparse(next_url).netloc and next_url not in visited:
                    queue.append(next_url)

            time.sleep(2)

        except Exception as e:
            print(f"Erreur: {e}")

    print(f"\nresultats: {len(matches)} images trouvée")
    return matches

if __name__ == "__main__":
    crawl_for_matching_images("./iris/src/scrapping/mwa.png", "https://www.superprof.fr/", max_pages=10000, tolerance_threshold=8)