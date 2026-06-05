import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

def download_one(idx, output_dir):
    url = f"https://picsum.photos/800/600?random={idx}"
    dest = os.path.join(output_dir, f"bg_{idx}.jpg")
    try:
        # Request with a user-agent to prevent blocks
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            with open(dest, 'wb') as f:
                f.write(response.read())
        if idx % 100 == 0:
            print(f"Downloaded {idx} / 1000 images.")
    except Exception as e:
        print(f"Failed to download image {idx}: {e}")

def main():
    output_dir = r"C:\Users\User\Desktop\TarotTrainingData\ImageNet"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting download of 1000 background images to {output_dir} using thread pool...")
    
    # Use 20 threads to speed up downloads
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(download_one, i, output_dir) for i in range(1000)]
        for f in futures:
            f.result()
            
    print("Download completed!")

if __name__ == "__main__":
    main()
