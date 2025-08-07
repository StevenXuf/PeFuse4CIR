import os
import requests
from PIL import Image
from io import BytesIO

from configuration import get_default_config

config = get_default_config()

output_dir = config['FashionIQ']['OUTPUT_DIR']
url_folder = config['FashionIQ']['IMAGE_URL_FOLDER']
# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

resize_to = (config['FashionIQ']['IMAGE_SIZE'], config['FashionIQ']['IMAGE_SIZE'])

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
