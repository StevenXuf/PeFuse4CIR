import sys
import torch

from torch.utils.data import DataLoader

from fashioniq import transform_image
from configuration import get_default_config

sys.path.append('/data/data_fxu/CIRCO/src')
from dataset import CIRCODataset

def get_circo_loader(batch_size=16, split='val', num_workers=0, transform=None):
    """
    Get DataLoader for CIRCO dataset.
    """
    dataset = CIRCODataset(data_path='/data/data_fxu/CIRCO', split=split, mode='relative')
    loader = DataLoader(dataset, 
                        batch_size=batch_size, 
                        shuffle=False, 
                        num_workers=num_workers,
                        collate_fn=lambda batch: {
                            'reference_img': torch.stack([transform(item['reference_img']) if transform is not None else item['reference_img'] for item in batch]),
                            'reference_pil': [item['reference_img'] for item in batch],
                            'reference_id': [item['reference_img_id'] for item in batch],
                            'target_img': torch.stack([transform(item['target_img']) if transform is not None else item['target_img'] for item in batch]),
                            'target_pil': [item['target_img'] for item in batch],
                            'target_id': [item['target_img_id'] for item in batch],
                            'caption': [item['relative_caption'] for item in batch],
                            'concept': [item['shared_concept'] for item in batch],
                            'all_target_img_ids': [item['gt_img_ids'] for item in batch],
                            'all_target_pil': [item for sublist in batch for item in sublist['gt_img']],
                            'all_target_img': torch.stack([transform(item) if transform is not None else item for sublist in batch for item in sublist['gt_img']]),
                            'all_target_length': list(map(len, [item['gt_img'] for item in batch])),
                            'query_id': [item['query_id'] for item in batch]
                        }
    )
    #[list(map(transform,item['gt_img']) if transform is not None else item['gt_img'] for item in batch)]

    return loader

if __name__ == "__main__":
    cfg = get_default_config("config.yaml")
    img_transform = transform_image(
        cfg['CLIP']['IMAGE_SIZE'],
        cfg['CLIP']['IMAGE_MEAN'],
        cfg['CLIP']['IMAGE_STD']
    )
    circo_loader = get_circo_loader(batch_size=cfg['GENERAL']['BATCH_SIZE'], split='val', num_workers=cfg['GENERAL']['NUM_WORKERS'], transform=img_transform)

    for batch in circo_loader:
        tar_pil = batch['all_target_length']
        print(tar_pil)