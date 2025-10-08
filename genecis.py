import os
from vaw_dataset import get_vaw_loader
from coco_dataset import get_coco_loader
from utils import transform_image, get_default_config

def get_genecis_loader(dataset_name, genecis_root, batch_size=16, num_workers=0, transform=None):
    if dataset_name.lower() == 'focus_attribute' or dataset_name.lower() == 'change_attribute':
        loader = get_vaw_loader(
            data_path=os.path.join(genecis_root, f'{dataset_name.lower()}.json'),
            batch_size=batch_size,
            num_workers=num_workers,
            transform=transform
            )
    elif dataset_name.lower() == 'focus_object' or dataset_name.lower() == 'change_object':
        loader = get_coco_loader(
            os.path.join(genecis_root,f'{dataset_name.lower()}.json'),
            batch_size=batch_size,
            num_workers=num_workers,
            transform=transform
            )
    else:
        raise ValueError('Invalid dataset name for GeneCIS')
    return loader


if __name__ == '__main__':
    cfg = get_default_config("config.yaml")
    img_transform = transform_image(
        cfg['CLIP']['IMAGE_SIZE'],
        cfg['CLIP']['IMAGE_MEAN'],
        cfg['CLIP']['IMAGE_STD']
    )
    loader = get_genecis_loader('change_attribute', '/data/data_fxu/GeneCIS/genecis/genecis',
                                transform=img_transform)
    for batch in loader:
        print(batch['reference_img'].shape)
        print(batch['caption'])
        break