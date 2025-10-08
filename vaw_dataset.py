# Copyright (c) Meta Platforms, Inc. and affiliates.

from torch.utils.data import Dataset, DataLoader

from PIL import Image
import json
import torch
import os

from utils import transform_image, get_default_config

genecis_root = '/data/data_fxu/GeneCIS/genecis/genecis'
visual_genome_images = '/data/data_fxu/GeneCIS/VG_100K'
coco_root = '/data/data_fxu/COCO/images/val2017'

DILATION = 0.7
PAD_CROP = True

def expand2square(pil_img, background_color=(0, 0, 0)):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

class VAWDataset(Dataset):

    def __init__(self, image_dir=visual_genome_images) -> None:
        super().__init__()

        self.image_dir = image_dir
        self.dilate = DILATION
        self.pad_crop = PAD_CROP

    def load_cropped_image(self, img):

        image_id = img['image_id']
        bbox = img['instance_bbox']
        
        # Get image
        path = os.path.join(self.image_dir, f'{image_id}.jpg')
        im = Image.open(path)
        im_width, im_height = im.size

        width = bbox[2]     # Width of bounding box
        height = bbox[3]    # Height of bounding box


        if self.dilate:
            orig_left, orig_top = bbox[0], bbox[1]
            left, top = max(0, orig_left - self.dilate * width), max(0, orig_top - self.dilate * height)
            right, bottom = min(im_width, left + (1 + self.dilate) * width), min(im_height, top + (1 + self.dilate) * height)
        else:
            left, top = bbox[0], bbox[1]
            right, bottom = bbox[0] + width, bbox[1] + height

        im = im.crop((left, top, right, bottom))
        
        if self.pad_crop:
            if im.mode == 'L':
                bg_color = (0,)
            else:
                bg_color = (0, 0, 0)
            im = expand2square(im, bg_color)

        return im.convert('RGB').resize((224, 224), Image.Resampling.BICUBIC)

class VAWValSubset(VAWDataset):

    def __init__(self, val_split_path, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        with open(val_split_path) as f:
            val_samples = json.load(f)

        self.val_samples = val_samples

    def __getitem__(self, index):
        
        """
        Follow same return signature as CIRRSubset
            (Except for returning reference object at the end)
        """

        sample = self.val_samples[index]
        reference = sample['reference']

        target = sample['target']
        gallery = sample['gallery']
        caption = sample['condition']

        reference, target = [self.load_cropped_image(i) for i in (reference, target)]
        gallery = [self.load_cropped_image(i) for i in gallery]

        gallery_and_target = [target] + gallery

        # By construction, target_rank = 0
        return {'reference_image': reference, 
                'caption': caption, 
                'target_images': gallery_and_target, 
                'target_id': list(map(lambda x: x['image_id'], [sample['target']]+sample['gallery'])),
                'target_rank': 0}
    

    def __len__(self):
        return len(self.val_samples)

def get_vaw_loader(data_path, batch_size=16, num_workers=0, transform=None):
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
    dataset = VAWValSubset(
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

    loader = get_vaw_loader(
        os.path.join(genecis_root, 'focus_attribute.json'), # attribute
        transform=img_transform
        )
    print(len(loader.dataset))
    for batch in loader:
        print(batch["reference_img"].shape, batch["target_img"].shape)
        print(batch["caption"])
        break