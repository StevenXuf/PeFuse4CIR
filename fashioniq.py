import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from refinedfashioniq import transform_image
from configuration import get_default_config

class FashionIQDataset(Dataset):
    def __init__(self, data_path, split='val', caption_folder='captions', split_folder='image_splits', mode='relative', transform=None):
        self.data_path = data_path
        self.image_path = os.path.join(data_path, "downloaded_images")
        self.split = split
        self.mode = mode
        self.caption_folder = os.path.join(data_path, caption_folder)
        self.split_folder = os.path.join(data_path, split_folder)
        self.transform = transform
        self.triplets, self.candidates = self.get_triplets()
        self.length = list(map(len, self.triplets))
        self.candidate_length = list(map(len, self.candidates))
        self.triplets = [item for sublist in self.triplets for item in sublist]
        self.candidates = [item for sublist in self.candidates for item in sublist]
        print(f"Loaded {len(self.triplets)} items from {self.split} split.")

    def get_triplets(self):
        # Load your dataset here
        captions = []
        candidates = []
        for cloth in ['shirt', 'dress', 'toptee']:
            existing_item=[]
            existing_candidates=[]
            if self.split.lower() =='test':
                with open(os.path.join(self.caption_folder, f"cap.{cloth}.{self.split.lower()}.json"), 'r') as f:
                    data = json.load(f)
                    for item in data:
                        if os.path.exists(os.path.join(self.image_path, item['candidate'] + '.jpg')):
                            existing_item.append(item)
            elif self.split.lower() =='train' or self.split.lower() == 'val':
                with open(os.path.join(self.caption_folder, f"cap.{cloth}.{self.split.lower()}.json"), 'r') as f:
                    data = json.load(f)
                    for item in data:
                        if os.path.exists(os.path.join(self.image_path, item['target'] + '.jpg')) and os.path.exists(os.path.join(self.image_path, item['candidate'] + '.jpg')):
                            existing_item.append(item)
            else:
                raise ValueError(f"{self.image_path} does not exist")
            captions.append(existing_item)
            
            with open(os.path.join(self.split_folder, f"split.{cloth}.{self.split.lower()}.json"), 'r') as f:
                candidates_data = json.load(f)
                for candidate in candidates_data:
                    if os.path.exists(os.path.join(self.image_path, candidate + '.jpg')):
                        existing_candidates.append(candidate)
            candidates.append(existing_candidates)
        return captions, candidates

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.candidates)

    def __getitem__(self, idx):
        if self.mode == 'classic':
            item = self.candidates[idx]
            image = Image.open(os.path.join(self.image_path, item + '.jpg')).convert('RGB').resize((224, 224), Image.Resampling.BICUBIC)
            if self.transform:
                image = self.transform(image)
            return {
                "query_id": str(idx),
                "image": image,
                "image_id": item,
            }

        elif self.mode == 'relative':
            item = self.triplets[idx]
            if self.split.lower() == 'train' or self.split.lower() == 'val':
                target_image = Image.open(os.path.join(self.image_path, item['target'] + '.jpg')).convert('RGB').resize((224, 224), Image.Resampling.BICUBIC)
                reference_image = Image.open(os.path.join(self.image_path, item['candidate'] + '.jpg')).convert('RGB').resize((224, 224), Image.Resampling.BICUBIC)

                if self.transform:
                    target_image = self.transform(target_image)
                    reference_image = self.transform(reference_image)
                return {
                    "query_id": str(idx),
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
                    "query_id": str(idx),
                    "caption": f"{' and '.join(item['captions'])}",
                    "reference_image": reference_image,
                    "reference_id": item["candidate"],
                }
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

def get_fashioniq_loader(data_path, batch_size=16, split='val', num_workers=0, mode='relative', transform=None):
    collate_fn_val_train=lambda batch: {
                            "query_id": [item["query_id"] for item in batch],
                            "caption": [item["caption"] for item in batch],
                            "reference_img": torch.stack([transform(item["reference_image"]) for item in batch]),
                            "reference_pil": [item["reference_image"] for item in batch],
                            "reference_id": [item["reference_id"] for item in batch],
                            "target_img": torch.stack([transform(item["target_image"]) for item in batch]),
                            "target_pil": [item["target_image"] for item in batch],
                            "target_id": [item["target_id"] for item in batch],
                            "target_length": list(map(len, [[item["target_image"]] for item in batch]))
                        }
    collate_fn_test=lambda batch: {
                            "query_id": [item["query_id"] for item in batch],
                            "caption": [item["caption"] for item in batch],
                            "reference_img": torch.stack([transform(item["reference_image"]) for item in batch]),
                            "reference_pil": [item["reference_image"] for item in batch],
                            "reference_id": [item["reference_id"] for item in batch],
                        }
    collate_fn_classic=lambda batch: {
                            "query_id": [item["query_id"] for item in batch],
                            "target_img": torch.stack([transform(item["image"]) for item in batch]),
                            "target_pil": [item["image"] for item in batch],
                            "target_id": [item["image_id"] for item in batch],
                            "target_length": list(map(len, [[item["image"]] for item in batch]))
                        }
    dataset = FashionIQDataset(data_path, 
                               split=split,
                               mode=mode
                               )
    if mode == 'relative':
        collate_fn = collate_fn_val_train if split in ['train', 'val'] else collate_fn_test
    elif mode == 'classic':
        collate_fn = collate_fn_classic
    else:
        raise ValueError(f"Unknown mode: {mode}")
    loader = DataLoader(dataset, 
                        batch_size=batch_size, 
                        num_workers=num_workers,
                        shuffle=False,
                        collate_fn=collate_fn
    )
    return loader


if __name__ == "__main__":
    cfg = get_default_config('config.yaml')

    img_transform = transform_image(
        cfg['CLIP']['IMAGE_SIZE'],
        cfg['CLIP']['IMAGE_MEAN'],
        cfg['CLIP']['IMAGE_STD']
    )
    loader = get_fashioniq_loader(cfg['FashionIQ']['IMAGE_FOLDER'], transform=img_transform, split='val', mode='classic')
    print(loader.dataset.length)
    print(loader.dataset.candidate_length)
    for batch in loader:
        print(batch["target_id"])
        print(batch["target_img"].size())
        break