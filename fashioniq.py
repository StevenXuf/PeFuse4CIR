import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from refinedfashioniq import transform_image
from configuration import get_default_config

class FashionIQDataset(Dataset):
    def __init__(self, data_path, split='val', caption_folder='captions', transform=None):
        self.data_path = data_path
        self.image_path = os.path.join(data_path, "downloaded_images")
        self.split = split
        self.caption_folder = os.path.join(data_path, caption_folder)
        self.transform = transform
        self.caption = self.load_captions()
        self.length = list(map(len, self.caption))
        self.caption = [item for sublist in self.load_captions() for item in sublist]
        print(f"Loaded {len(self.caption)} items from {self.split} split.")

    def load_captions(self):
        # Load your dataset here
        captions = []
        for cloth in ['dress', 'shirt', 'toptee']:
            existing_item=[]
            with open(os.path.join(self.caption_folder, f"cap.{cloth}.{self.split}.json"), 'r') as f:
                data = json.load(f)
                for item in data:
                    if os.path.exists(os.path.join(self.image_path, item['target'] + '.jpg')) and os.path.exists(os.path.join(self.image_path, item['candidate'] + '.jpg')):
                        existing_item.append(item)
            captions.append(existing_item)
        return captions

    def __len__(self):
        return len(self.caption)

    def __getitem__(self, idx):
        item = self.caption[idx]
        target_image = Image.open(os.path.join(self.image_path, item['target'] + '.jpg')).convert('RGB')
        reference_image = Image.open(os.path.join(self.image_path, item['candidate'] + '.jpg')).convert('RGB')

        if self.transform:
            target_image = self.transform(target_image)
            reference_image = self.transform(reference_image)

        return {
            "caption": item["captions"][0],
            "reference_image": reference_image,
            "target_image": target_image,
            "reference_id": item["candidate"],
            "target_id": item["target"]
        }

def get_fashioniq_loader(data_path, batch_size=16, split='val', num_workers=0, transform=None):
    dataset = FashionIQDataset(data_path, 
                               split=split
                               )
    loader = DataLoader(dataset, 
                        batch_size=batch_size, 
                        num_workers=num_workers,
                        shuffle=True,
                        collate_fn=lambda batch: {
                            "caption": [item["caption"] for item in batch],
                            "reference_img": torch.stack([transform(item["reference_image"]) for item in batch]),
                            "reference_pil": [item["reference_image"] for item in batch],
                            "reference_id": [item["reference_id"] for item in batch],
                            "target_img": torch.stack([transform(item["target_image"]) for item in batch]),
                            "target_pil": [item["target_image"] for item in batch],
                            "target_id": [item["target_id"] for item in batch],
                            "all_target_img": torch.stack([transform(item["target_image"]) for item in batch]),
                            "all_target_id": [item["target_id"] for item in batch],
                            "all_target_pil": [item["target_image"] for item in batch],
                            "all_target_id": [item["target_id"] for item in batch],
                            "all_target_length": list(map(len, [[item["target_image"]] for item in batch]))
                        }
    )
    return loader


if __name__ == "__main__":
    cfg = get_default_config('config.yaml')

    img_transform = transform_image(
        cfg['CLIP']['IMAGE_SIZE'],
        cfg['CLIP']['IMAGE_MEAN'],
        cfg['CLIP']['IMAGE_STD']
    )
    loader = get_fashioniq_loader(cfg['FashionIQ']['IMAGE_FOLDER'], transform=img_transform)
    print(loader.dataset.length)
    for batch in loader:
        print(batch["caption"])
        print(batch["reference_img"].shape, batch["target_img"].shape)
        break