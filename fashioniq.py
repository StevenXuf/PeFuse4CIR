import os
import requests
import json
import torch

from tqdm import tqdm
from pathlib import Path
from PIL import Image
from io import BytesIO
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

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


def extract_candidates_and_captions(json_path, split='test'):
    directory = Path(json_path)
    matching_files = list(directory.rglob(f'*{split}*.json'))
    print(matching_files)
    for json_file in matching_files:
        # Load the JSON data from a file
        with open(json_file, 'r') as file:
            data = json.load(file)

        # Extract candidate and captions
        for item in data:
            print(item)
            candidate = item.get("candidate")
            captions = item.get("captions", [])
            
            print(f"Candidate: {candidate}")
            for caption in captions:
                print(f"  - {caption}")

def transform_image():
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    img_transform = transforms.Compose([
        transforms.CenterCrop(224),  # Standard size for most CNNs
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return img_transform

def get_fashioniq_dataset(output_dir,batch_size=32,transform=None):
    """
    Load the FashionIQ dataset.
    """
    dataset = load_dataset("chuonghm/Refined-FashionIQ", split='validation')
    dataset = dataset.filter(lambda x: x['is_refined'] == True)

    target_images=[]
    reference_images=[]
    captions=[]
    cnt=0

    for item in tqdm(dataset):
        try:
            target = Image.open(os.path.join(output_dir, f'{item["target"]}.jpg')).convert('RGB')
            reference = Image.open(os.path.join(output_dir, f'{item["candidate"]}.jpg')).convert('RGB')
            target_images.append(target)
            reference_images.append(reference)
            captions.append(item['captions'])
        except Exception as e:
            cnt+=1
            print(f"Error processing {cnt}th item: {e}")
            continue

    dataset = Dataset.from_dict({
        'target': target_images,
        'reference': reference_images,
        'caption': captions
    })
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
        collate_fn=lambda batch: {
            'target': torch.stack([x['target'] if img_transform is None else img_transform(x['target']) for x in batch]),
            'reference': torch.stack([x['reference'] if img_transform is None else img_transform(x['reference']) for x in batch]),
            'caption': [x['caption'] for x in batch]}
            )

    return dataloader

if __name__ == "__main__":
    config = get_default_config()

    output_dir = config['FashionIQ']['OUTPUT_DIR']
    url_folder = config['FashionIQ']['IMAGE_URL_FOLDER']
    resize_to = (config['FashionIQ']['IMAGE_SIZE'], config['FashionIQ']['IMAGE_SIZE'])
    batch_size = config['FashionIQ']['BATCH_SIZE']

    img_transform = transform_image()

    if not os.path.exists(output_dir):
        download_and_resize_images(output_dir, url_folder, resize_to)
    else:
        print(f"Output directory already exists in: {output_dir}")

    # extract_candidates_and_captions(config['FashionIQ']['CAPTION_FOLDER'],split='test')
    dataloader = get_fashioniq_dataset(output_dir,transform=img_transform, batch_size=batch_size)
    print(f"Loaded FashionIQ dataset with {len(dataloader.dataset)} items.")
    for batch in dataloader:
        print(f"Batch size: {len(batch['target'])}")
        print(f"Target images shape: {batch['target'].shape}")
        print(f"Reference images shape: {batch['reference'].shape}")
        print(f"Captions: {batch['caption'][:1]}")