import requests
import os
import bs4

def download_images(query, num_images):
    query = '+'.join(query.split())
    url = f'https://www.google.com/search?hl=en&tbm=isch&q={query}&tbs=sur:f'


    res = requests.get(url)
    res.raise_for_status()

    folder_name = f"{query.replace('+', '_')}Images"
    os.makedirs(folder_name, exist_ok=True)

    soup = bs4.BeautifulSoup(res.text, 'html.parser')
    all_images = soup.select('img[src]')

    count = 0
    for post in all_images:
        src = post.get('src')
        if not src:
            print('Could not find image source.')
            continue

        try:
            image_res = requests.get(src)
            image_res.raise_for_status()
            print(f'Downloading {src} to folder {folder_name}...')

            image_file_path = os.path.join(folder_name, f'{query}_{count + 1}.jpg')
            with open(image_file_path, 'wb') as image_file:
                for chunk in image_res.iter_content(100000):
                    image_file.write(chunk)

            count += 1
            if count >= num_images:
                break

        except Exception as e:
            print(f'Failed to download {src}. Reason: {e}')

    print('Done downloading images.')

download_images('Person driving motorcycle on roadway from back', 20)
