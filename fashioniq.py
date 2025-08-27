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
        for cloth in ['shirt', 'dress', 'toptee']:
            existing_item=[]
            if self.split.lower() =='test':
                with open(os.path.join(self.caption_folder, f"cap.{cloth}.test.json"), 'r') as f:
                    data = json.load(f)
                    for item in data:
                        if os.path.exists(os.path.join(self.image_path, item['candidate'] + '.jpg')):
                            existing_item.append(item)
            elif self.split.lower() =='train' or self.split.lower() == 'val':
                with open(os.path.join(self.caption_folder, f"cap.{cloth}.{self.split}.json"), 'r') as f:
                    data = json.load(f)
                    for item in data:
                        if os.path.exists(os.path.join(self.image_path, item['target'] + '.jpg')) and os.path.exists(os.path.join(self.image_path, item['candidate'] + '.jpg')):
                            existing_item.append(item)
            else:
                raise ValueError(f"{self.image_path} does not exist")
            captions.append(existing_item)
        return captions

    def __len__(self):
        return len(self.caption)

    def __getitem__(self, idx):
        item = self.caption[idx]
        if self.split.lower() == 'train' or self.split.lower() == 'val':
            target_image = Image.open(os.path.join(self.image_path, item['target'] + '.jpg')).convert('RGB').resize((224, 224), Image.Resampling.BICUBIC)
            reference_image = Image.open(os.path.join(self.image_path, item['candidate'] + '.jpg')).convert('RGB').resize((224, 224), Image.Resampling.BICUBIC)

            if self.transform:
                target_image = self.transform(target_image)
                reference_image = self.transform(reference_image)
            return {
                "caption": f"{' and '.join(item['captions'])}",
                "reference_image": reference_image,
                "target_image": target_image,
                "reference_id": item["candidate"],
                "target_id": item["target"]
            }
        else:
            reference_image = Image.open(os.path.join(self.image_path, item['candidate'] + '.jpg')).convert('RGB').resize((224, 224), Image.Resampling.BICUBIC)
            if self.transform:
                reference_image = self.transform(reference_image)
            return {
                "caption": f"{' and '.join(item['captions'])}",
                "reference_image": reference_image,
                "reference_id": item["candidate"],
            }

def get_fashioniq_loader(data_path, batch_size=16, split='val', num_workers=0, transform=None):
    collate_fn_val_train=lambda batch: {
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
    collate_fn_test=lambda batch: {
                            "caption": [item["caption"] for item in batch],
                            "reference_img": torch.stack([transform(item["reference_image"]) for item in batch]),
                            "reference_pil": [item["reference_image"] for item in batch],
                            "reference_id": [item["reference_id"] for item in batch],
                        }
    dataset = FashionIQDataset(data_path, 
                               split=split
                               )
    loader = DataLoader(dataset, 
                        batch_size=batch_size, 
                        num_workers=num_workers,
                        shuffle=True,
                        collate_fn=collate_fn_val_train if split in ['train', 'val'] else collate_fn_test
    )
    return loader


if __name__ == "__main__":
    cfg = get_default_config('config.yaml')

    img_transform = transform_image(
        cfg['CLIP']['IMAGE_SIZE'],
        cfg['CLIP']['IMAGE_MEAN'],
        cfg['CLIP']['IMAGE_STD']
    )
    loader = get_fashioniq_loader(cfg['FashionIQ']['IMAGE_FOLDER'], transform=img_transform, split='val')
    print(loader.dataset.length)
    for batch in loader:
        print(batch["caption"])
        print(batch["reference_img"].shape)
        break