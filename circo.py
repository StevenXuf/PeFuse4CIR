import torch
import json
import os
from PIL import Image

from pathlib import Path
from typing import Union, List, Dict, Literal
from torch.utils.data import Dataset, DataLoader

from refinedfashioniq import transform_image
from configuration import get_default_config

class CIRCODataset(Dataset):
    """
    CIRCO dataset
    """

    def __init__(self, data_path: Union[str, Path], split: Literal['val', 'test'], mode: Literal['relative', 'classic'],
                 preprocess=None):
        """
        Args:
            data_path (Union[str, Path]): path to CIRCO dataset
            split (str): dataset split, should be in ['test', 'val']
            mode (str): dataset mode, should be in ['relative', 'classic']
            preprocess (callable): function which preprocesses the image
        """

        # Set dataset paths and configurations
        data_path = Path(data_path)
        self.mode = mode
        self.split = split
        self.preprocess = preprocess
        self.data_path = data_path

        # Ensure input arguments are valid
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")
        if split not in ['test', 'val']:
            raise ValueError("split should be in ['test', 'val']")

        # Load COCO images information
        with open(data_path / 'COCO2017_unlabeled' / "annotations" / "image_info_unlabeled2017.json", "r") as f:
            imgs_info = json.load(f)

        self.img_paths = [data_path / 'COCO2017_unlabeled' / "unlabeled2017" / img_info["file_name"] for img_info in
                          imgs_info["images"]]
        self.img_ids = [img_info["id"] for img_info in imgs_info["images"]]
        self.img_ids_indexes_map = {str(img_id): i for i, img_id in enumerate(self.img_ids)}

        # get CIRCO annotations
        with open(data_path / 'annotations' / f'{split}.json', "r") as f:
            self.annotations: List[dict] = json.load(f)

        # Get maximum number of ground truth images (for padding when loading the images)
        self.max_num_gts = 23  # Maximum number of ground truth images

        print(f"CIRCODataset {split} dataset in {mode} mode initialized")


    def __getitem__(self, index) -> dict:
        """
        Returns a specific item from the dataset based on the index.

        In 'classic' mode, the dataset yields a dictionary with the following keys: [img, img_id]
        In 'relative' mode, the dataset yields dictionaries with the following keys:
            - [reference_img, reference_img_id, target_img, target_img_id, relative_caption, shared_concept, gt_img_ids,
            query_id] if split == val
            - [reference_img, reference_img_id, relative_caption, shared_concept, query_id]  if split == test
        """

        if self.mode == 'relative':
            # Get the query id
            query_id = str(self.annotations[index]['id'])

            # Get relative caption and shared concept
            relative_caption = self.annotations[index]['relative_caption']
            shared_concept = self.annotations[index]['shared_concept']

            # Get the reference image
            reference_img_id = str(self.annotations[index]['reference_img_id'])
            reference_img_path = self.img_paths[self.img_ids_indexes_map[reference_img_id]]
            reference_img = self.preprocess(Image.open(reference_img_path).convert('RGB')) if self.preprocess is not None else Image.open(reference_img_path).convert('RGB')

            if self.split == 'val':
                # Get the target image and ground truth images
                target_img_id = str(self.annotations[index]['target_img_id'])
                gt_img_ids = [str(x) for x in self.annotations[index]['gt_img_ids']]
                target_img_path = self.img_paths[self.img_ids_indexes_map[target_img_id]]
                target_img = self.preprocess(Image.open(target_img_path).convert('RGB')) if self.preprocess is not None else Image.open(target_img_path).convert('RGB')
                # Pad ground truth image IDs with zeros for collate_fn
                gt_img_ids += [''] * (self.max_num_gts - len(gt_img_ids))
                gt_img_ids = [ gt_img_id for gt_img_id in gt_img_ids if len(gt_img_id) > 0]  
                gt_img_paths = [os.path.join(self.data_path,'COCO2017_unlabeled/unlabeled2017', tar_id.zfill(12) + '.jpg') for tar_id in gt_img_ids]
                gt_img = [self.preprocess(Image.open(gt_path).convert('RGB')) if self.preprocess is not None else Image.open(gt_path).convert('RGB') for gt_path in gt_img_paths]

                return {
                    'reference_img': reference_img,
                    'reference_img_id': reference_img_id,
                    'target_img': target_img,
                    'target_img_id': target_img_id,
                    'relative_caption': relative_caption,
                    'shared_concept': shared_concept,
                    'gt_img_ids': gt_img_ids,
                    'gt_img': gt_img,
                    'query_id': query_id,
                }

            elif self.split == 'test':
                return {
                    'reference_img': reference_img,
                    'reference_img_id': reference_img_id,
                    'relative_caption': relative_caption,
                    'shared_concept': shared_concept,
                    'query_id': query_id,
                }

        elif self.mode == 'classic':
            # Get image ID and image path
            img_id = str(self.img_ids[index])
            img_path = self.img_paths[index]

            # Preprocess image and return
            img = self.preprocess(Image.open(img_path).convert('RGB'))if self.preprocess is not None else Image.open(img_path).convert('RGB')
            return {
                'img': img,
                'img_id': img_id
            }

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        if self.mode == 'relative':
            return len(self.annotations)
        elif self.mode == 'classic':
            return len(self.img_ids)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")


def get_circo_loader(data_path, batch_size=16, split='val', num_workers=0, transform=None):
    """
    Get DataLoader for CIRCO dataset.
    """
    val_collate_fn = lambda batch:{
        'reference_img': torch.stack([transform(item['reference_img']) if transform is not None else item['reference_img'] for item in batch]),
        'reference_pil': [item['reference_img'] for item in batch],
        'reference_id': [item['reference_img_id'] for item in batch],
        'target_img': torch.stack([transform(item['target_img']) if transform is not None else item['target_img'] for item in batch]),
        'target_pil': [item['target_img'] for item in batch],
        'target_id': [item['target_img_id'] for item in batch],
        'caption': [f"{item['relative_caption']} and has {item['shared_concept']}." for item in batch],
        'concept': [item['shared_concept'] for item in batch],
        'all_target_ids': [item['gt_img_ids'] for item in batch],
        'all_target_pil': [item for sublist in batch for item in sublist['gt_img']],
        'all_target_img': torch.stack([transform(item) if transform is not None else item for sublist in batch for item in sublist['gt_img']]),
        'all_target_length': list(map(len, [item['gt_img'] for item in batch])),
        'query_id': [item['query_id'] for item in batch]
    }
    test_collate_fn = lambda batch:{
        'reference_img': torch.stack([transform(item['reference_img']) if transform is not None else item['reference_img'] for item in batch]),
        'reference_pil': [item['reference_img'] for item in batch],
        'reference_id': [item['reference_img_id'] for item in batch],
        'caption': [f"{item['relative_caption']} while {item['shared_concept']} should be shown." for item in batch],
        'concept': [item['shared_concept'] for item in batch],
        'query_id': [item['query_id'] for item in batch]
    }

    dataset = CIRCODataset(data_path=data_path, split=split, mode='relative')
    loader = DataLoader(dataset, 
                        batch_size=batch_size, 
                        shuffle=False, 
                        num_workers=num_workers,
                        collate_fn=test_collate_fn if split == 'test' else val_collate_fn
    )

    return loader


if __name__ == "__main__":
    cfg = get_default_config("config.yaml")
    img_transform = transform_image(
        cfg['CLIP']['IMAGE_SIZE'],
        cfg['CLIP']['IMAGE_MEAN'],
        cfg['CLIP']['IMAGE_STD']
    )
    circo_loader = get_circo_loader(cfg['CIRCO']['IMAGE_FOLDER'], batch_size=cfg['GENERAL']['BATCH_SIZE'], split='val', num_workers=cfg['GENERAL']['NUM_WORKERS'], transform=img_transform)

    for batch in circo_loader:
        ref_pil = batch['reference_img']
        tar_pil = batch['target_img']
        print(batch['caption'])
        break
        # Just to test the loader