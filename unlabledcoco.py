import torch
import os

from PIL import Image
from torch.utils.data import DataLoader, Dataset

from feature_extraction import transform_image
from configuration import get_default_config

class UnlabeledCOCO(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.image_path = os.path.join(data_path, "COCO2017_unlabeled/unlabeled2017")
        self.images = os.listdir(self.image_path)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_id = self.images[idx]
        img_path = os.path.join(self.image_path, img_id)

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {"image": image, "image_id": img_id.split('.')[0]}


def get_unlabeledcoco_loader(image_path, batch_size=16, transform=None, num_workers=0):
    """
    Load all unlabeled images from the dataset.
    """
    dataset = UnlabeledCOCO(data_path=image_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            collate_fn=lambda batch:{
                                "image": torch.stack([transform(item["image"]) for item in batch]),
                                "image_id": [item["image_id"] for item in batch]
                            }
                            )
    return dataloader

if __name__ == "__main__":
    cfg= get_default_config('./config.yaml')
    image_path = cfg['CIRCO']['IMAGE_FOLDER']
    transform = transform_image(
        cfg['CLIP']['IMAGE_SIZE'],
        cfg['CLIP']['IMAGE_MEAN'],
        cfg['CLIP']['IMAGE_STD']
    )
    dataloader = get_unlabeledcoco_loader(image_path, batch_size=16, transform=transform, num_workers=0)
    for batch in dataloader:
        images = batch["image"]
        image_ids = batch["image_id"]
        print(f"Image size: {images.size()}, Image IDs: {image_ids}")
        break