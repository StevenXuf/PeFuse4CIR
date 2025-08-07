import os
import requests
import json

from pathlib import Path
from PIL import Image
from io import BytesIO

from configuration import get_default_config

def download_and_resize_images(output_dir, url_folder, resize_to=(224, 224)):
    """
    Download images from URLs in specified text files, resize them, and save to output directory.
    """
    print("Starting image download and resizing...")

    # Check if the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Settings
    for file in ['asin2url.dress.txt', 'asin2url.shirt.txt', 'asin2url.toptee.txt']:
        input_file = f'{os.path.join(url_folder, file)}'  # path to your .txt file

        # Read and process each line
        with open(input_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue  # skip empty lines
                try:
                    image_id, image_url = line.split()

                    # Download image
                    response = requests.get(image_url, timeout=10)
                    response.raise_for_status()  # raise error for bad status

                    # Open image
                    image = Image.open(BytesIO(response.content)).convert('RGB')

                    # Resize image
                    image = image.resize(resize_to, Image.Resampling.LANCZOS)

                    # Save image
                    output_path = os.path.join(output_dir, f"{image_id}.jpg")
                    image.save(output_path)
                    print(f"Downloaded and resized: {image_id}")
                except Exception as e:
                    print(f"Failed to process line: {line}\nError: {e}")
        print(f"Finished processing {input_file}")
    print("All done!")


def extract_candidates_and_captions(json_file,split='test'):
    directory = Path(json_file).parent
    matching_files = list(directory.rglob(f'*{split}*.json'))
    for json_file in matching_files:
        # Load the JSON data from a file
        with open(json_file, 'r') as file:
            data = json.load(file)

        # Extract candidate and captions
        for item in data:
            candidate = item.get("candidate")
            captions = item.get("captions", [])
            
            print(f"Candidate: {candidate}")
            for caption in captions:
                print(f"  - {caption}")

if __name__ == "__main__":
    config = get_default_config()

    output_dir = config['FashionIQ']['OUTPUT_DIR']
    url_folder = config['FashionIQ']['IMAGE_URL_FOLDER']
    resize_to = (config['FashionIQ']['IMAGE_SIZE'], config['FashionIQ']['IMAGE_SIZE'])

    download_and_resize_images(output_dir, url_folder, resize_to)
    extract_candidates_and_captions(config['FashionIQ']['CAPTION_FOLDER'],split='test')
