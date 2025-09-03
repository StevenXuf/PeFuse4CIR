import torch
import os
import json

from PIL import Image
from torch.utils.data import DataLoader, Dataset

from utils import transform_image
from configuration import get_default_config

class TestImageDataset(Dataset):
    def __init__(self, data_name, transform=None):
        self.cfg = get_default_config("config.yaml")
        self.data_name = data_name
        if self.data_name.lower() == 'circo':
            self.image_folder = os.path.join(self.cfg['CIRCO']['IMAGE_FOLDER'], "COCO2017_unlabeled/unlabeled2017")
            self.images = os.listdir(self.image_folder)
        elif self.data_name.lower() == 'cirr':
            self.image_folder = os.path.join(self.cfg['CIRR']['IMAGE_FOLDER'], "img_raw", "test1")
            with open(os.path.join(self.cfg['CIRR']['IMAGE_FOLDER'], "image_splits", f"split.rc2.test1.json")) as f:
                data = json.load(f)
            self.images = list(map(lambda x: f"{x}.png", data.keys()))
        self.transform = transform
        print(f"Number of items in images: {len(self.images)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_id = self.images[idx]
        img_path = os.path.join(self.image_folder, img_id)

        image = Image.open(img_path).convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)

        if self.transform:
            image = self.transform(image)

        return {
            "target_image": image,
            "target_id": img_id.split('.')[0]
            }


def get_test_image_loader(dataset_name, batch_size=16, transform=None, num_workers=0):
    """
    Load all test images from the dataset.
    """
    dataset = TestImageDataset(data_name=dataset_name)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            collate_fn=lambda batch:{
                                "target_img": torch.stack([transform(item["target_image"]) if transform else item["target_image"] for item in batch]),
                                "target_pil": [item["target_image"] for item in batch],
                                "target_id": [item["target_id"] for item in batch],
                                "target_length": list(map(len,[[item["target_image"]] for item in batch]))
                            }
                            )
    return dataloader

if __name__ == "__main__":
    cfg = get_default_config("config.yaml")
    transform = transform_image(
        cfg['CLIP']['IMAGE_SIZE'],
        cfg['CLIP']['IMAGE_MEAN'],
        cfg['CLIP']['IMAGE_STD']
    )
    dataloader = get_test_image_loader('circo', batch_size=512, transform=transform, num_workers=0)
    for batch in dataloader:
        images = batch["target_img"]
        image_ids = batch["target_id"]
        print(f"Image size: {images.size()}")
        break