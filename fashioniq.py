import os
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from refinedfashioniq import transform_image
from configuration import get_default_config

class FashionIQDataset(Dataset):
    def __init__(self, data_path, split='val', transform=None):
        self.data_path = data_path
        self.split = split
        self.transform = transform
        self.data = self.load_data()

    def load_data(self):
        # Load your dataset here
        return []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if self.transform:
            item = self.transform(item)
        return item

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
                            "reference_img": [item["reference_img"] for item in batch],
                            "target_img": [item["target_img"] for item in batch]
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
    for batch in loader:
        print(batch["caption"])
        print(batch["reference_img"].shape, batch["target_img"].shape)
        break