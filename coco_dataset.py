# Copyright (c) Meta Platforms, Inc. and affiliates.

from torch.utils.data import Dataset, DataLoader
import os

from PIL import Image
import json
import torch

from utils import transform_image, get_default_config

coco_root = '/data/data_fxu/COCO/images/val2017'
genecis_root = '/data/data_fxu/GeneCIS/genecis/genecis'   


class COCODataset(Dataset):

    def __init__(self, root_dir=coco_root) -> None:
        super().__init__()

        self.root_dir = root_dir

    def load_sample(self, sample):

        val_img_id = sample['val_image_id']
        fpath = os.path.join(self.root_dir, f'{val_img_id:012d}.jpg')
        img = Image.open(fpath).convert('RGB').resize((224, 224), Image.Resampling.BICUBIC)
        
        return img

class COCOValSubset(COCODataset):

    def __init__(self, val_split_path, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        with open(val_split_path) as f:
            val_samples = json.load(f)

        self.val_samples = val_samples

    def __getitem__(self, index):
        
        """
        Follow same return signature as CIRRSubset
        """

        sample = self.val_samples[index]
        reference = sample['reference']

        target = sample['target']
        gallery = sample['gallery']
        caption = sample['condition']

        reference, target = [self.load_sample(i) for i in (reference, target)]
        gallery = [self.load_sample(i) for i in gallery]

        gallery_and_target = [target] + gallery

        # By construction, target_rank = 0
        return {'reference_image': reference, 
                'caption': caption, 
                'target_images': gallery_and_target, 
                'target_id': list(map(lambda x: x['val_image_id'], [sample['target']]+sample['gallery'])),
                'target_rank': 0}

    def __len__(self):
        return len(self.val_samples)


def get_coco_loader(data_path, batch_size=16, num_workers=0, transform=None):
    collate_fn = lambda batch: {
        'reference_img': torch.stack([transform(item['reference_image']) if transform is not None else item['reference_image'] for item in batch]),
        'reference_pil': [item['reference_image'] for item in batch],
        'caption': [item['caption'] for item in batch],
        'target_img': torch.stack([transform(tgt) if transform is not None else tgt for item in batch for tgt in item['target_images']]),
        'target_pil': [tgt for item in batch for tgt in item['target_images']],
        'target_length': list(map(len, [item['target_images'] for item in batch])),
        'target_id': [item['target_id'] for item in batch],
        'query_id': []
    }

    dataset = COCOValSubset(
        val_split_path=data_path
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate_fn
    )
    return loader

if __name__ == '__main__':
    cfg = get_default_config("config.yaml")
    img_transform = transform_image(
        cfg['CLIP']['IMAGE_SIZE'],
        cfg['CLIP']['IMAGE_MEAN'],
        cfg['CLIP']['IMAGE_STD']
    )
    
    loader = get_coco_loader(
        os.path.join(genecis_root, 'focus_object.json'),
        transform=img_transform
        )
    print(len(loader.dataset))
    for batch in loader:
        print(batch["reference_img"].shape, batch["target_img"].shape)
        print(batch["caption"])
        break