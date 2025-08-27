import os
import requests
import torch

from tqdm import tqdm
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
            for line in tqdm(f, desc=f"Processing {input_file}"):
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
                    # image = image.resize(resize_to, Image.Resampling.LANCZOS)

                    # Save image
                    output_path = os.path.join(output_dir, f"{image_id}.jpg")
                    image.save(output_path)
                    print(f"Downloaded: {image_id}")
                except Exception as e:
                    print(f"Failed to process line: {line}\nError: {e}")
        print(f"Finished processing {input_file}")
    print("All done!")

def transform_image(image_size, IMAGENET_MEAN=None, IMAGENET_STD=None):
    img_transform = transforms.Compose([
        transforms.Resize((image_size, image_size),interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD) if IMAGENET_MEAN is not None and IMAGENET_STD is not None else transforms.Lambda(lambda x: x)
    ])
    return img_transform

def get_refined_fashioniq_loader(output_dir,batch_size=32,transform=None):
    """
    Load the RefinedFashionIQ dataset.
    """
    dataset = load_dataset("chuonghm/Refined-FashionIQ", split='validation')
    dataset = dataset.filter(lambda x: x['is_refined'] == True)

    target_images = []
    target_ids = []
    reference_images = []
    reference_ids = []
    captions = []
    cnt = 0

    for item in tqdm(dataset):
        try:
            target = Image.open(os.path.join(output_dir, f'{item["target"]}.jpg')).resize((224, 224), Image.Resampling.BICUBIC)
            reference = Image.open(os.path.join(output_dir, f'{item["candidate"]}.jpg')).resize((224, 224), Image.Resampling.BICUBIC)
            target_images.append(target)
            reference_images.append(reference)
            captions.append(item['captions'][0])
            target_ids.append(item['target'])
            reference_ids.append(item['candidate'])
        except Exception as e:
            cnt += 1
            print(f"Error processing {cnt}th item: {e}")
            continue

    dataset = Dataset.from_dict({
        'target': target_images,
        'target_id': target_ids,
        'reference': reference_images,
        'reference_id': reference_ids,
        'caption': captions
    })
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
        collate_fn=lambda batch: {
            'target_img': torch.stack([x['target'] if transform is None else transform(x['target']) for x in batch]),
            'target_id': [x['target_id'] for x in batch],
            'reference_img': torch.stack([x['reference'] if transform is None else transform(x['reference']) for x in batch]),
            'reference_id': [x['reference_id'] for x in batch],
            'caption': [x['caption'] for x in batch],
            'target_pil': [x['target'] for x in batch],
            'reference_pil': [x['reference'] for x in batch],
            'all_target_pil': [x['target'] for x in batch],
            'all_target_img': torch.stack([x['target'] if transform is None else transform(x['target']) for x in batch]),
            'all_target_length': list(map(len, [[x['target']] for x in batch])),
        })

    return dataloader

if __name__ == "__main__":
    config = get_default_config()

    output_dir = config['RefinedFashionIQ']['IMAGE_FOLDER']
    url_folder = config['RefinedFashionIQ']['IMAGE_URL_FOLDER']
    resize_to = (config['RefinedFashionIQ']['IMAGE_SIZE'], config['RefinedFashionIQ']['IMAGE_SIZE'])
    batch_size = config['GENERAL']['BATCH_SIZE']
    mean = config['CLIP']['IMAGE_MEAN']
    std = config['CLIP']['IMAGE_STD']  

    img_transform = transform_image(config['RefinedFashionIQ']['IMAGE_SIZE'], mean, std)

    if not os.path.exists(output_dir):
        download_and_resize_images(output_dir, url_folder, resize_to)
    else:
        print(f"Output directory already exists in: {output_dir}")

    dataloader = get_refined_fashioniq_loader(output_dir,transform=img_transform, batch_size=batch_size)
    print(f"Loaded RefinedFashionIQ dataset with {len(dataloader.dataset)} items.")
    for batch in dataloader:
        print(f"Batch size: {len(batch['target_pil'])}")
        print(f"Target images shape: {batch['target_img'][0].shape}")
        print(f"Reference images shape: {batch['reference_img'][0].shape}")
        print(f"Captions: {batch['caption']}")